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
        num_docs = len(self.tokenized_docs)
        self.idf = {word: math.log(num_docs / (freq + 1)) + 1 
                    for word, freq in self.doc_freqs.items()}
        self.embedding_matrix = self._fit_transform()

    def get_embedding(self):
        if self.embedding_matrix is None:
            self.fit()
        return self.embedding_matrix

    def _fit_transform(self):
        num_docs = len(self.tokenized_docs)
        batch_size = 1000  # 한 번에 처리할 문서 수
        embed_list = []

        for start in tqdm(range(0, num_docs, batch_size), desc='Calculating TF-IDF'):
            end = min(start + batch_size, num_docs)
            batch_docs = self.tokenized_docs[start:end]
            batch_embed = self._process_batch(batch_docs)
            embed_list.append(batch_embed)

        print('Finish TF-IDF Embedding')
        return vstack(embed_list)

    def _process_batch(self, batch_docs):
        batch_size = len(batch_docs)
        batch_embed = np.zeros((batch_size, len(self.vocab)), dtype=np.float32)

        for i, doc in enumerate(batch_docs):
            doc_ngrams = self._get_ngrams(doc)
            doc_tf = Counter(doc_ngrams)
            doc_len = len(doc_ngrams)
            for ngram, freq in doc_tf.items():
                if ngram in self.vocab:
                    idx = self.vocab[ngram]
                    tf = freq / doc_len
                    batch_embed[i, idx] = tf * self.idf[ngram]

        return csr_matrix(batch_embed)

    def _get_ngrams(self, tokens):
        n_grams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            n_grams.extend([' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])
        return n_grams

    def transform(self, tokenized_query):
        query_ngrams = self._get_ngrams(tokenized_query)
        query_vector = np.zeros(len(self.vocab), dtype=np.float32)
        query_tf = Counter(query_ngrams)
        query_len = len(query_ngrams)
        for ngram, freq in query_tf.items():
            if ngram in self.vocab:
                idx = self.vocab[ngram]
                tf = freq / query_len
                query_vector[idx] = tf * self.idf[ngram]
        return csr_matrix(query_vector)