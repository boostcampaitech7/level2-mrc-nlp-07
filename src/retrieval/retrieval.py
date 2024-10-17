import json
import os
import pickle
from typing import List, NoReturn, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
from scipy.sparse import save_npz, load_npz, vstack
from src.utils.timer import timer
from src.embedding.sparse_embedding import SparseEmbedding

class SparseRetrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:

        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
        """
        # wiki를 불러오기 위해 path 결합 및 불러오기.
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        # wiki 데이터 불러오고 contexts와 id 저장, contexts - > key가 context이고, value는 None으로 저장
        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])  # [1,2,3] -> fromkeys - > dict{1:None, 2:None, 3:None}
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))
        self.tokenize_fn = tokenize_fn
        # Transform by vectorizer
        self.tfidfv = TfidfVectorizer(
            tokenizer=tokenize_fn, ngram_range=(1, 2), max_features=50000, # 사용할 최대 빈도수를 50,000개로 제한.
        )
        # get_sparse_embedding()로 생성합니다
        self.p_embedding = None  
        self.mode = None 
        self.sparse_embed = None 
        
    def get_sparse_embedding(self, mode: str = 'tfidf') -> NoReturn:
        """
        Summary:
            선택된 mode에 따라 Passage Embedding을 생성하고 저장합니다.
            이미 저장된 파일이 있다면 해당 파일을 불러옵니다.
        
        Args:
            mode (str): 임베딩 방법 선택
                - 'tfidf': scikit-learn의 TF-IDF
                - 'my_tfidf': 직접 구현한 TF-IDF
                - 'bm25': 직접 구현한 BM25
        """
        self.mode = mode
        
        pickle_name = f"{mode}_embedding.npz"
        sparse_name = f"{mode}_vector.bin"
        emd_path = os.path.join(self.data_path, pickle_name)
        sparse_path = os.path.join(self.data_path, sparse_name)
        print(f"Building {mode} embedding...")
        # SparseEmbedding 객체 생성 (모드를 지정하여 필요한 임베딩만 계산)
        self.sparse_embed = SparseEmbedding(
            corpus = self.contexts, 
            tokenizer = self.tokenize_fn, 
            mode = mode)
        
        # 시간이 오래걸려서 저장 여부 선택.
        if os.path.isfile(emd_path) and os.path.isfile(sparse_path):
            with open(sparse_path, "rb") as file:
                self.tfidfv = pickle.load(file)
            self.sparse_embed.load(sparse_path)
        else:
            self.p_embedding = self.sparse_embed.get_embedding() 
            save_npz(emd_path, self.p_embedding)
            self.sparse_embed.save(sparse_path)

        
        print("Embedding Finished")
        print(f"{mode} embedding shape:", self.p_embedding.shape)
        print()

    # FAISS (Facebook AI Similarity Search) 라이브러리를 사용하여 벡터 인덱스 구성
    # 벡터 인덱스 구성 이유?

    # 유사도 검색을 통한 비슷한 문서 검색
    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1 # Union은 주로 여러개의 typing 형이 가능할 때 주로 가용. Optinal은 1값이나 int형 값을 가질 수 있다는 의미.
    ) -> Union[Tuple[List, List], pd.DataFrame]: # 이 retrive의 결과로 튜플 형식의 (list, list)를 반환하거나 데이터프레임을 반환.

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

        assert self.p_embedding is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."
        # 위에서 p_embedding이 제대로 불러와졌는지 확인.
        # query or dataset이 string타입의 경우
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk) # 가장 유사도가 높은 k개의 query 또는 dataset 반환
            print("[Search query]\n", query_or_dataset, "\n")
            # k개 만큼의 결과 출력
            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])
        # query or dataset이 dataset이 아닐경우 -> 이는 쿼리가 한개가 아니라 여러개라는 의미.
        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            # 위에서 선언한 contextmanager 들고와서 걸리는 시간 Check!
            with timer("query exhaustive search"): # 
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            # 쿼리와 ID와 내용 저장.
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "retrieval_context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]] # 현재 쿼리에 대한 가장 유사도가 높은 k개의 content들을 합침.
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    # 검증 데이터에는 아래와 같이 분리되어 잇음.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total) # contex, question, answer, 
            return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        # 위에서 선언한 contextmanager 들고와서 걸리는 시간 Check!
        with timer("transform"):
            query_vec = self.tfidfv.transform([query]) # 쿼리문을 벡터화
        # 쿼리문이 정상적으로 바뀌였는지 확인
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."
        
        # 쿼리 벡터와 
        with timer("query ex search"):
            result = query_vec * self.p_embedding.T # (1, 50,000) x (50,000 x 문서 벡터 수) -> (1, 문서 수) -> 가장 유사한 문서를 찾을 수 있다..?
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        # 1차원 벡터로 전환후 내림 차순으로 정렬 -> 0번째의 인덱스가 가장 유사한 문서.
        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k] # result를 1차원 벡터 -> 이를 위헤서 내림차순으로 정렬 -> 리스트로 변환 -> 상위 탑 k개만큼 slicing
        doc_indices = sorted_result.tolist()[:k] # 상위 k에 대한 인덱스 슬라이싱
        return doc_score, doc_indices 

    def get_relevant_doc_bulk(self, queries: List, k: Optional[int] = 1) -> Tuple[List, List]:
        """
        Arguments:
            queries (List):
                여러 개의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab에 없는 이상한 단어로 query하는 경우 assertion 발생 (예) 뙣뙇?
        """
        if self.mode is None:
            raise ValueError("Embedding mode not set. Call get_sparse_embedding first.")

        # Query vector 계산
        query_vecs = vstack([self.sparse_embed.transform(query) for query in queries])
        print(query_vecs.shape)
        assert (
            np.sum(query_vecs) != 0
        ), "query_vecs가 제대로 변환되지않음."

        print('유사도 계산')
        print(query_vecs.shape, self.p_embedding.shape)

        result = query_vecs @ self.p_embedding.T  # 행렬 곱 연산

        print('유사도 후')
        print(f'result shape : {result.shape}')

        if not isinstance(result, np.ndarray):
            result = result.toarray()

        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])

        return doc_scores, doc_indices