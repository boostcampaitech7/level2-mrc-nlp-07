from datasets import concatenate_datasets, load_from_disk
from src.retriever.retrieval.sparse_retrieval import SparseRetrieval
from transformers import AutoTokenizer
from src.config.path_config import TRAIN_DATA_PATH, TEST_DATA_PATH, SAVE_TRAIN_PATH, SAVE_TEST_PATH


tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
retriever = SparseRetrieval(
    tokenize_fn=tokenizer.tokenize,
    data_path="./data/",
    context_path="filtered_wiki.json",
    mode = "bm25",
    max_feature=1000000,
    ngram_range=(1,2),
)


if __name__ == '__main__':
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
