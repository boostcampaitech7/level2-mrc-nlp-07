from __future__ import annotations

import ast

from reader.model.reader import Reader
from transformers import HfArgumentParser
from transformers import TrainingArguments
from utils.arguments import DataTrainingArguments
from utils.arguments import ModelArguments
from datasets import concatenate_datasets, load_from_disk
from src.retriever.retrieval.sparse_retrieval import SparseRetrieval
from transformers import AutoTokenizer
from src.config.path_config import *
from datasets import load_dataset, DatasetDict
from src.config import key_names

def retriever():
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    retriever = SparseRetrieval(
        tokenize_fn=tokenizer.tokenize,
        data_path= DATA_PATH,
        context_path= WIKI_PATH,
        mode = "bm25",
        max_feature=1000000,
        ngram_range=(1,2),
    )
    org_dataset = load_from_disk(TRAIN_DATA_PATH)
    full_ds = concatenate_datasets(
            [
                org_dataset["train"].flatten_indices(),
                org_dataset["validation"].flatten_indices(),
            ]
    )
    print("*" * 40, "Train dataset", "*" * 40)
    print(full_ds)
    test_dataset = load_from_disk(TEST_DATA_PATH)
    test_dataset = test_dataset['validation'].flatten_indices()
    print(test_dataset)
    retriever.get_sparse_embedding()
    train_df = retriever.retrieve(full_ds, topk=10)
    train_df.to_csv(SAVE_TRAIN_PATH)
    print("*" * 40,'Save Train Result',"*" * 40)
    retriever.get_score(train_df)
    test_df = retriever.retrieve(test_dataset, topk=10)
    test_df.to_csv(SAVE_TEST_PATH)
    print("*" * 40,'Save Test Result',"*" * 40)
    print("*" * 40,'Finish Retriever',"*" * 40)


def reader():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # CSV 파일에서 데이터 로드
    train_retriever_dataset = load_dataset('csv', data_files=SAVE_TEST_PATH)

    # DatasetDict로 train과 validation을 정의
    train_dataset_dict = DatasetDict({
        key_names.TRAIN: train_retriever_dataset[key_names.TRAIN],  # 'train' 스플릿으로 지정
        key_names.VALIDATION: train_retriever_dataset[key_names.VALIDATION]  # 'validation' 스플릿으로 지정
    })

    # 불필요한 컬럼 제거 및 'retrieval_context'를 'context'로 변경
    train_dataset_dict[key_names.TRAIN] = train_dataset_dict[key_names.TRAIN]\
        .remove_columns(key_names.REMOVE_COLUMNS_FROM_RETRIEVER)\
        .rename_column(key_names.RETRIEVAL_CONTEXT, key_names.CONTEXT)

    # 'answers' 필드를 파싱하여 딕셔너리로 변환하는 함수
    def process_answers(example):
        # 'answers' 필드가 문자열로 저장된 경우 이를 딕셔너리로 변환
        if isinstance(example[key_names.ANSWER], str):
            example[key_names.ANSWER] = ast.literal_eval(example[key_names.ANSWER])
        return example

    # 'answers' 필드를 처리하여 파싱
    train_dataset_dict[key_names.TRAIN] = train_dataset_dict[key_names.TRAIN]\
    .map(process_answers)


    # output_dir은 training_args에 설정된 값을 사용
    print('Output directory:', training_args.output_dir)

    # Argument 설정
    model_args = ModelArguments()
    data_args = DataTrainingArguments()
    training_args = TrainingArguments(output_dir=outputs)

    # 학습/평가/추론 설정
    training_args.do_train = True
    training_args.do_eval = False
    training_args.do_predict = False

    # 학습 하이퍼파라미터 설정
    model_args.model_name_or_path = 'klue/bert-base'  # 학습에 사용할 모델
    training_args.learning_rate = 5e-5  # 학습률
    training_args.num_train_epochs = 1  # epoch 수
    training_args.per_device_train_batch_size = 16  # 학습 배치 사이즈

    reader_model = Reader(model_args, data_args, training_args, train_dataset_dict)
    reader_model.run()


if __name__ == '__main__':
    retriever()
    reader()
