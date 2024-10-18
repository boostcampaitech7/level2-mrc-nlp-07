import math
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
from tqdm import tqdm


class Tfidf:
    def __init__(self, vocab:dict, doc_freqs, tokenized_docs:list, ngram_range:tuple, max_features:int):
        self.vocab = vocab
        self.doc_freqs = doc_freqs
        self.tokenized_docs = tokenized_docs
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.embedding_matrix = None
        self.idf = None

    def fit(self):
        self.embedding_matrix = self._fit_transform()

    def get_embedding(self):
        if self.embedding_matrix is None:
            self.fit()
        return self.embedding_matrix

    def _fit_transform(self):
        num_docs = len(self.tokenized_docs)
        self.idf = {word: math.log(
            num_docs / (freq + 1)) + 1 for word, freq in self.doc_freqs.items()}

        batch_size = 1000  # 한 번에 처리할 문서 수
        embed_list = []

        for i in tqdm(range(0, num_docs, batch_size), desc='Calculating TF-IDF'):
            batch_docs = self.tokenized_docs[i:i+batch_size]
            batch_embed = self._process_batch(batch_docs)
            embed_list.append(csr_matrix(batch_embed))

        print('Finish TF-IDF Embedding')
        return vstack(embed_list)

    def _process_batch(self, batch_docs):
        batch_embed = np.zeros(
            (len(batch_docs), len(self.vocab)), dtype=np.float32)

        for doc_idx, doc in enumerate(batch_docs):
            doc_tf = Counter(doc)
            doc_len = len(doc)
            for word, freq in doc_tf.items():
                if word in self.vocab:
                    idx = self.vocab[word]
                    batch_embed[doc_idx, idx] = (
                        freq / doc_len) * self.idf[word]

        return batch_embed

    def transform(self, tokenized_query: str):
        query_vector = np.zeros(len(self.vocab), dtype=np.float32)
        for word in tokenized_query:
            if word in self.vocab:
                idx = self.vocab[word]
                query_vector[idx] = 1
        return csr_matrix(query_vector)
