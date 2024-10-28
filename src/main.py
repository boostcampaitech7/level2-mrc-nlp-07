from __future__ import annotations

from reader.model.reader import Reader
from transformers import HfArgumentParser
from transformers import TrainingArguments
from utils.arguments import DataTrainingArguments
from utils.arguments import ModelArguments
from datasets import concatenate_datasets, load_from_disk
from src.retriever.retrieval.sparse_retrieval import SparseRetrieval
from transformers import AutoTokenizer
from src.config.path_config import *

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

    # output_dir은 training_args에 설정된 값을 사용
    print('Output directory:', training_args.output_dir)

    reader_model = Reader(model_args, data_args, training_args)
    reader_model.run()


if __name__ == '__main__':
    retriever()
    reader()
