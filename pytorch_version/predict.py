import time
import os

import torch

from models.albert_for_ner import AlbertCrfForNer
from models.bert_for_ner import BertCrfForNer
from models.transformers import BertConfig, AlbertConfig
from processors.ner_seq import ner_processors as processors
from processors.utils_ner import CNerTokenizer, get_entities
from tools.common import init_logger, logger
from tools.common import seed_everything

_project_dir = os.path.dirname(__file__)
output_dir = f'{_project_dir}/outputs/skillner_output/bert'
task_name = "skillner"
seed = 42
do_lower_case = True
eval_max_seq_length = 512
markup = 'bios'
model_name_or_path = f'{_project_dir}/prev_trained_model/roberta_wwm_large_ext'
model_type = 'bert'

processor = processors[task_name]()
label_list = processor.get_labels()
num_labels = len(label_list)
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else 'cpu')


device = get_device()


def convert_to_features(example,
                        max_seq_length,
                        tokenizer,
                        cls_token_at_end=False,
                        cls_token="[CLS]",
                        cls_token_segment_id=1,
                        sep_token="[SEP]",
                        pad_on_left=False,
                        pad_token=0,
                        pad_token_segment_id=0,
                        sequence_a_segment_id=0,
                        mask_padding_with_zero=True, ):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    tokens = tokenizer.tokenize(example)
    # Account for [CLS] and [SEP] with "- 2".
    special_tokens_count = 2
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[: (max_seq_length - special_tokens_count)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens += [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)

    if cls_token_at_end:
        tokens += [cls_token]
        segment_ids += [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
    input_len = len(tokens)
    # Zero-pad up to the sequence length.
    #     padding_length = max_seq_length - len(input_ids)
    padding_length = 0
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
    else:
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length

    #     assert len(input_ids) == max_seq_length
    #     assert len(input_mask) == max_seq_length
    #     assert len(segment_ids) == max_seq_length

    logger.info("*** Example ***")
    logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
    logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
    logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
    logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))

    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long).to(device),
        "attention_mask": torch.tensor([input_mask], dtype=torch.long).to(device),
        "labels": None,
        "token_type_ids": torch.tensor([segment_ids], dtype=torch.long).to(device),
        'input_lens': torch.tensor([input_len], dtype=torch.long).to(device)
    }


def load_model():
    global model_type
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    init_logger(log_file=output_dir + f'/{model_type}-{task_name}-{time_}.log')
    logger.warning("Process device: %s,16-bits training", device)
    # Set seed
    seed_everything(seed)
    # Prepare NER task

    assert task_name in processors

    model_type = model_type.lower()
    config_class, model_class, tokenizer_class = BertConfig, BertCrfForNer, CNerTokenizer
    config = config_class.from_pretrained(model_name_or_path, num_labels=num_labels, cache_dir=None)
    _tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=do_lower_case, cache_dir=None)
    _model = model_class.from_pretrained(output_dir, from_tf=False, config=config, cache_dir=None)
    _model.to(device)
    _model.eval()
    return _tokenizer, _model


tokenizer, model = load_model()


def predict(text):
    inputs = convert_to_features(
        text,
        max_seq_length=eval_max_seq_length,
        tokenizer=tokenizer,
        cls_token_at_end=bool(model_type in ["xlnet"]),
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
        sep_token=tokenizer.sep_token,
        pad_on_left=bool(model_type in ["xlnet"]),
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if model_type in ['xlnet'] else 0,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True
    )
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs[0]
        tags = model.crf.decode(logits, inputs['attention_mask'])
        tags = tags.squeeze(0).cpu().numpy().tolist()
    preds = tags[0][1:-1]  # [CLS]XXXX[SEP]
    label_entities = get_entities(preds, id2label, markup)
    for [ner, start, end] in label_entities:
        yield ner, text[start:end + 1]
