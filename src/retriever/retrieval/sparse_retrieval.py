import json
import os
from typing import List, NoReturn, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm
from scipy.sparse import save_npz, load_npz, vstack
from src.utils.timer import timer
from src.retriever.embedding.sparse_embedding import SparseEmbedding
from src.retriever.score.ranking import check_original_in_context, calculate_reverse_rank_score, calculate_linear_score
import re
from scipy.spatial.distance import cdist


class SparseRetrieval:
    def __init__(
        self,
        tokenize_fn= None,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
        mode : Optional[str] = 'bm25',
        max_feature:int = 1000000,
        ngram_range :tuple = (1,2),
        tokenized_docs = None,
        k1: float = 1.1,
        b: float = 0.5,
    ) -> NoReturn:
        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수
                아래와 같은 함수들을 사용할 수 있음.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로

            context_path:
                Passage들이 묶여있는 파일명
            
            mode:
                'tfidf' : sklearn의 tfidf 모드. 'my_tfidf' : 직접 구현한 tfidf 모드. 'bm25' : 직접 구현한 bm25 모드.
            max_feature : Embedding 차원 int형 타입을 받으며, default = 1000000,
            ngram_range : 단어장 생성시 n_gram을 얼마나 할지 결정, default = (1,2),
        Summary:
            Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
        """
        # wiki를 불러오기 위해 path 결합 및 불러오기.
        self.data_path = data_path
        self.tokenize_fn = tokenize_fn
        self.ngram_range = ngram_range
        self.max_features = max_feature
        self.p_embedding = None
        self.mode = mode
        self.sparse_embed = None
        self.pickle_name = f"{mode}_embedding.npz"
        self.sparse_name = f"{mode}_vector.bin"
        if max_feature is not None:
            self.pickle_name = f"{mode}_embedding_{str(max_feature)}.npz"
            self.sparse_name = f"{mode}_vector_{str(max_feature)}.bin"
        self.emd_path = os.path.join(self.data_path, self.pickle_name)
        self.sparse_path = os.path.join(self.data_path, self.sparse_name)
        self.tokenized_docs = tokenized_docs
        
        self.k1, self.b = k1, b
        self._initialize_from_wiki(context_path)
    
    def clean_text(self, text):
        # 특수 문자 제거
        text = re.sub(r'[^\w\s\.\,\?\!]', ' ', text)
        # 불필요한 패턴 제거
        patterns_to_remove = [
            r'[\'\"\[\]\{\}\(\)]',  # 따옴표, 괄호류
            r'\'text\':|\'context\':', # 'text':, 'context': 패턴
            r'[a-zA-Z]\)',  # (a), (b) 같은 패턴
            r'\\[a-zA-Z]',  # \x, \u 등의 escape sequence
        ]
        for pattern in patterns_to_remove:
            text = re.sub(pattern, ' ', text)
        escape_sequences = [
        r'\\',       # \
        r'\/n',      # /n
        r'\/',       # /
        r':',        # : 
        ]
        for escape in escape_sequences:
            text = text.replace(escape, ' ')
        # 한글과 숫자만 영어 남기고 모두 제거
        cleaned_text = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text)
        # 여러 개의 공백을 하나의 공백으로 변경
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        return cleaned_text.strip()
        
    def _initialize_from_wiki(self, context_path: str):
        with open(os.path.join(self.data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)
        
        self.original_docs = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        self.cleaned_docs = [self.clean_text(doc) for doc in self.original_docs]
    # 원본 텍스트와 정제된 텍스트의 매핑 사전 생성
        print(f"Lengths of unique contexts : {len(self.original_docs)}")
        #self.ids = list(range(len(self.docs)))
        
    def get_sparse_embedding(self) -> NoReturn:
        """
        Summary:
            선택된 mode에 따라 Passage Embedding을 생성하고 저장합니다.
            이미 저장된 파일이 있다면 해당 파일을 불러옵니다.
        
        Args:
            mode (str): 임베딩 방법 선택
                - 'tfidf': scikit-learn의 TF-IDF
                - 'our_tfidf': 직접 구현한 TF-IDF
                - 'bm25': 직접 구현한 BM25
        """
        if os.path.isfile(self.emd_path) and os.path.isfile(self.sparse_path):
            print(f"Loading {self.mode} embedding...")
            self.p_embedding = load_npz(self.emd_path)
            self.sparse_embed = SparseEmbedding.load(self.sparse_path)
            print("Loading completed.")
        else:
            print(f"Building {self.mode} embedding...")
            self._calculate_embeddings()
        print(f"{self.mode} embedding shape:", self.p_embedding.shape)
        
    def _calculate_embeddings(self):
        self.sparse_embed = SparseEmbedding(
            docs=self.cleaned_docs,
            tokenizer=self.tokenize_fn,
            mode= self.mode,
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            k1 = self.k1,
            b = self.b,
        )
        self.p_embedding = self.sparse_embed.get_embedding()
        save_npz(self.emd_path, self.p_embedding)
        self.sparse_embed.save(self.sparse_path)
        print("New embeddings calculated and saved.")

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
                print(self.original_docs[doc_indices[i]])

            return (doc_scores, [self.original_docs[doc_indices[i]] for i in range(topk)])
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
                    "retrieval_context": 
                        [self.original_docs[pid] for pid in doc_indices[idx]] # 현재 쿼리에 대한 가장 유사도가 높은 k개의 content들을 합침.
                    ,
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
            query = self.clean_text(query)
            query_vec = self.sparse_embed.transform([query]) # 쿼리문을 벡터화
        # 쿼리문이 정상적으로 바뀌였는지 확인
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."
        
        # 쿼리 벡터와 
        with timer("query ex search"):                                          # (위키)
            result = query_vec @ self.p_embedding.T # (1, 50,000) x (50,000 x 문서 벡터 수) -> (1, 문서 수) -> 가장 유사한 문서를 찾을 수 있다..?
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        # 1차원 벡터로 전환후 내림 차순으로 정렬 -> 0번째의 인덱스가 가장 유사한 문서.
        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k] # result를 1차원 벡터 -> 이를 위헤서 내림차순으로 정렬 -> 리스트로 변환 -> 상위 탑 k개만큼 slicing
        doc_indices = sorted_result.tolist()[:k] # 상위 k에 대한 인덱스 슬라이싱
        return doc_score, doc_indices 

    def get_relevant_doc_bulk(
        self, 
        queries: List, 
        k: Optional[int] = 5
        ) -> Tuple[List, List]:
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
        #cleaned_queries = [self.clean_text(q) for q in queries]
        stage1 = [self.sparse_embed.transform(query) for query in queries]
        query_vecs = vstack(stage1) # 질문수, 임베딩 차원
        assert (
            np.sum(query_vecs) != 0
        ), "query_vecs가 제대로 변환되지않음."

        print(query_vecs.shape, self.p_embedding.shape)
        # 유사도 계산
        
        result = query_vecs @ self.p_embedding.T  # 행렬 곱 연산 (질문수, 임베딩 차원) x (임베딩 차원, 문서수)
        # 질문수, 문서 수 -> 점수높은순으로 top-k
        print(f'result shape : {result.shape}')

        if not isinstance(result, np.ndarray):
            result = result.toarray()

        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1] # 높은순으로 인덱스 반환
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])

        return doc_scores, doc_indices
    
    def get_score(self, df):
        """
        Summary:
            Retriever 결과에 대해 출력하는 함수로, Correct는 정답이 Retrieval 결과에 포함됬는지 나타내는 갑이며,
            Reverse_Rank는 정답 인덱스의 역수로 0이면 1, 1이면 0.5의 값을 가진다.
            linear는 Reverse rank를 역수가 아닌 각 index마다 간격을 동일하게 주어
        """
        df["correct"] = df.apply(check_original_in_context, axis=1)
        df["rmm_score"] = df.apply(calculate_reverse_rank_score, axis=1)
        df["linear_score"] = df.apply(calculate_linear_score, axis=1)
        print(
            "correct retrieval:",
            round(df["correct"].sum() / len(df)),
        )
        print(
            "reverse rank retrieval:",
            round(df["rmm_score"].sum() / len(df))
        )
        print(
            "linear retrieval:",
            round(df["linear_score"].sum() / len(df))
        )
        total_true_count = df['correct'].sum()
        total_count = len(df)
        total_false_count = total_count - total_true_count
        total_true_ratio = (total_true_count / total_count) * 100
        total_false_ratio = (total_false_count / total_count) * 100
        
        print("=== 전체 데이터셋 통계 ===")
        print(f"전체 질문 수: {total_count}")
        print(f"정답 수 (True): {total_true_count} ({total_true_ratio:.2f}%)")
        print(f"오답 수 (False): {total_false_count} ({total_false_ratio:.2f}%)")
    
        print("\n=== 영어 포함 질문 통계 ===")
        english_questions = df[df['question'].str.contains('[a-zA-Z]', regex=True)]
        true_count = english_questions['correct'].sum()
        total_count = len(english_questions)
        false_count = total_count - true_count
        true_ratio = (true_count / total_count) * 100
        false_ratio = (false_count / total_count) * 100
        print(f"영어가 포함된 질문 수: {total_count}")
        print(f"정답 수 (True): {true_count} ({true_ratio:.2f}%)")
        print(f"오답 수 (False): {false_count} ({false_ratio:.2f}%)")
        
    def compute_l2_distance(query_vec, passage_vec) -> np.ndarray:
        """ 
        Arguments:
            query_vec:
                embedding된 query vector 입니다.

            passage_vec:
                embedding된 passage vector 입니다.

        Summary:
            query vector와 passage vector를 input으로 받고, L2 거리를 계산해주는 함수입니다.
        """

        # dense matrix로 변경
        if not isinstance(query_vec, np.ndarray):
            query_vec = query_vec.toarray()
        if not isinstance(passage_vec, np.ndarray):
            passage_vec = passage_vec.toarray()

        # 결과값 저장을 위한 빈 리스트 생성
        num_queries = query_vec.shape[0]
        num_passages = passage_vec.shape[0]
        l2_distances = np.zeros((num_queries, num_passages))

        for i in tqdm(range(num_queries), desc="Computing L2 distances"):
            # i번째 쿼리와 모든 passage 간의 L2 거리를 계산하고 저장
            l2_distances[i] = cdist(query_vec[i:i+1], passage_vec, metric='euclidean').flatten()
            print(f"L2 Distance: {l2_distances[i]}")

        print(f"L2_distances shape: ")
        return l2_distances