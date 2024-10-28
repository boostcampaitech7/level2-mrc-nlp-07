from datasets import concatenate_datasets, load_from_disk
from src.retriever.retrieval.sparse_retrieval import SparseRetrieval
from transformers import AutoTokenizer


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
    org_dataset = load_from_disk('./data/train_dataset')
    print(org_dataset)
    full_ds = concatenate_datasets(
            [
                org_dataset["train"].flatten_indices(),
                org_dataset["validation"].flatten_indices(),
            ]
    )
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)
    retriever.get_sparse_embedding()