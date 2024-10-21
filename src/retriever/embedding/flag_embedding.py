import os
import json
from typing import List, NoReturn, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm
from FlagEmbedding import BGEM3FlagModel
from sklearn.metrics.pairwise import cosine_similarity

class DenseRetrieval:
    def __init__(
        self,
        data_path: Optional[str] = "./data/",
        context_path: Optional[str] = "wikipedia_documents.json",
        model_name: str = "BAAI/bge-m3",
    ) -> NoReturn:

        self.data_path = data_path
        self.model_name = model_name
        self.model = BGEM3FlagModel(model_name, use_fp16=True)  # FlagEmbedding 모델 초기화
        self.p_embedding = None
        
        self._initialize_from_wiki(context_path)

    def _initialize_from_wiki(self, context_path: str):
        with open(os.path.join(self.data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.docs = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        print(f"Lengths of unique contexts : {len(self.docs)}")
        self.ids = list(range(len(self.docs)))

    def get_dense_embedding(self) -> NoReturn:
        print("Building dense embedding...")
        self.p_embedding = self.model.encode(self.docs, show_progress_bar=True)
        print("Dense embedding shape:", self.p_embedding.shape)

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        assert self.p_embedding is not None, "get_dense_embedding() 메소드를 먼저 수행해주세요."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")
            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.docs[doc_indices[i]])

            return (doc_scores, [self.docs[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            total = []
            with tqdm(total=len(query_or_dataset), desc="Dense retrieval: ") as pbar:
                for example in query_or_dataset:
                    doc_scores, doc_indices = self.get_relevant_doc(example["question"], k=topk)
                    tmp = {
                        "question": example["question"],
                        "id": example["id"],
                        "retrieval_context": " ".join(
                            [self.docs[pid] for pid in doc_indices]
                        ),
                    }
                    if "context" in example.keys() and "answers" in example.keys():
                        tmp["original_context"] = example["context"]
                        tmp["answers"] = example["answers"]
                    total.append(tmp)
                    pbar.update(1)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        query_vec = self.model.encode([query])[0]
        result = cosine_similarity(query_vec.reshape(1, -1), self.p_embedding)[0]
        sorted_result = np.argsort(result)[::-1]
        doc_score = result[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

    def get_relevant_doc_bulk(self, queries: List, k: Optional[int] = 1) -> Tuple[List, List]:
        query_vecs = self.model.encode(queries)
        result = cosine_similarity(query_vecs, self.p_embedding)
        
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])

        return doc_scores, doc_indices