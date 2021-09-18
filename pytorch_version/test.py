from processors.ner_seq import CluenerProcessor, SkillnerProcessor

if __name__ == '__main__':
    dataset_dir = "/Users/junix/code/CLUENER2020/pytorch_version/datasets/cluener"
    dataset_dir = '../../job-descriptor/skill-dataset'
    p = SkillnerProcessor()
    for x in p.get_train_examples(dataset_dir):
        print(x)
