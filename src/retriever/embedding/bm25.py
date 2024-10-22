from __future__ import annotations

import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
from tqdm import tqdm


class Bm25:
    def __init__(self, vocab, doc_freqs, tokenized_docs, ngram_range, max_features, k1: float = 1.1, b: float = 0.5):
        self.vocab = vocab
        self.doc_freqs = doc_freqs
        self.tokenized_docs = tokenized_docs
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.k1 = k1
        self.b = b
        self.doc_lengths = np.array([len(doc) for doc in self.tokenized_docs])
        self.average_document_length = np.mean(self.doc_lengths)
        self.idf = None
        self.embedding_matrix = None

    def fit(self):
        num_docs = len(self.tokenized_docs)
        self.idf = {
            word: math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1)
            for word, freq in self.doc_freqs.items()
        }
        self.embedding_matrix = self._fit_transform()

    def get_embedding(self):
        if self.embedding_matrix is None:
            self.fit()
        return self.embedding_matrix

    def _fit_transform(self):
        num_docs = len(self.tokenized_docs)
        batch_size = 1000  # 한 번에 처리할 문서 수
        embed_list = []
        for start in tqdm(range(0, num_docs, batch_size), desc='Calculating BM25'):
            end = min(start + batch_size, num_docs)
            batch_embed = self._process_batch(self.tokenized_docs[start:end], start)
            embed_list.append(batch_embed)

        print('Finish BM25 Embedding')
        return vstack(embed_list)

    def _process_batch(self, batch_docs, start_idx):
        batch_size = len(batch_docs)
        batch_embed = np.zeros((batch_size, len(self.vocab)), dtype=np.float32)

        for i, doc in enumerate(batch_docs):
            doc_ngrams = self._get_ngrams(doc)
            doc_len = self.doc_lengths[start_idx + i]
            for ngram in set(doc_ngrams):
                if ngram in self.vocab:
                    word_idx = self.vocab[ngram]
                    freq = doc_ngrams.count(ngram)
                    numerator = self.idf[ngram] * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.average_document_length)
                    batch_embed[i, word_idx] = numerator / denominator

        return csr_matrix(batch_embed)

    def _get_ngrams(self, tokens):
        n_grams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            n_grams.extend([' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])
        return n_grams

    def transform(self, tokenized_query: str):
        query_ngrams = self._get_ngrams(tokenized_query)
        query_vector = np.zeros(len(self.vocab), dtype=np.float32)
        for ngram in query_ngrams:
            if ngram in self.vocab:
                query_vector[self.vocab[ngram]] = 1
        return csr_matrix(query_vector)
