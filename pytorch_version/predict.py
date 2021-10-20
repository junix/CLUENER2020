import time

import torch

from models.albert_for_ner import AlbertCrfForNer
from models.bert_for_ner import BertCrfForNer
from models.transformers import BertConfig, AlbertConfig
from processors.ner_seq import ner_processors as processors
from processors.utils_ner import CNerTokenizer, get_entities
from tools.common import init_logger, logger
from tools.common import seed_everything

MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertConfig, BertCrfForNer, CNerTokenizer),
    'albert': (AlbertConfig, AlbertCrfForNer, CNerTokenizer)
}


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
    print(tokens)
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
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.tensor([input_mask], dtype=torch.long),
        "labels": None,
        "token_type_ids": torch.tensor([segment_ids], dtype=torch.long),
        'input_lens': torch.tensor([input_len], dtype=torch.long)
    }


output_dir = "outputs/skillner_output/bert/"
task_name = "skillner"
no_cuda = True
local_rank = -1
seed = 42

adam_epsilon = 1e-08
adv_epsilon = 1.0
adv_name = 'word_embeddings'
cache_dir = ''
config_name = ''
crf_learning_rate = 5e-05
data_dir = '/Users/junix/code/CLUENER2020/pytorch_version/datasets/skillner/'
do_adv = False
do_eval = True
do_lower_case = True
do_predict = False
do_train = False
eval_all_checkpoints = False
eval_max_seq_length = 512
evaluate_during_training = False
fp16 = False
fp16_opt_level = 'O1'
gradient_accumulation_steps = 1
learning_rate = 3e-05
local_rank = -1
logging_steps = 448
loss_type = 'ce'
markup = 'bios'
max_grad_norm = 1.0
max_steps = -1
model_name_or_path = '/Users/junix/code/CLUENER2020/pytorch_version/prev_trained_model/roberta_wwm_large_ext'
model_type = 'bert'
no_cuda = False
num_train_epochs = 5.0
output_dir = '/Users/junix/code/CLUENER2020/pytorch_version/outputs/skillner_output/bert'
overwrite_cache = False
overwrite_output_dir = True
per_gpu_eval_batch_size = 24
per_gpu_train_batch_size = 24
predict_checkpoints = 0
save_steps = 448
seed = 42
server_ip = ''
server_port = ''
task_name = 'skillner'
tokenizer_name = ''
train_max_seq_length = 128
warmup_proportion = 0.1
weight_decay = 0.01

processor = processors[task_name]()
label_list = processor.get_labels()
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}


def load_model():
    global model_type

    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    init_logger(log_file=output_dir + f'/{model_type}-{task_name}-{time_}.log')

    # Setup CUDA, GPU & distributed training
    if local_rank == -1 or no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1

    print(device, n_gpu)

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training",
        local_rank, device, n_gpu, bool(local_rank != -1), )
    # Set seed
    seed_everything(seed)
    # Prepare NER task

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    print(id2label)
    print(label2id)

    # Load pretrained model and tokenizer
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    model_type = model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(config_name if config_name else model_name_or_path,
                                          num_labels=num_labels, cache_dir=cache_dir if cache_dir else None, )
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name if tokenizer_name else model_name_or_path,
                                                do_lower_case=do_lower_case,
                                                cache_dir=cache_dir if cache_dir else None, )
    print("model_name_or_path>>", model_name_or_path)
    model = model_class.from_pretrained(output_dir, from_tf=bool(".ckpt" in model_name_or_path),
                                        config=config, cache_dir=cache_dir if cache_dir else None)
    if local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(device)
    logger.info("Training/evaluation parameters")
    model.eval()

    print(tokenizer)
    return tokenizer, model


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
