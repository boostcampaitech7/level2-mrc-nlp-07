import math
from collections import Counter
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import csr_matrix #   
import pickle

class SparseEmbedding:
    def __init__(self, corpus: List[str], tokenizer=None, ngram_range=(1,2), max_features:int=50000, mode: str = 'tfidf'):
        self.corpus = corpus
        self.tokenizer = tokenizer if tokenizer else lambda x: x.split()
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.mode = mode
        self.tokenized_corpus = [self.tokenizer(doc) for doc in self.corpus]
        self.avgdl = sum(len(doc) for doc in self.tokenized_corpus) / len(self.tokenized_corpus)
        
        self.vocab = self._build_vocab()
        self.doc_freqs = self._compute_doc_freqs()
        
        if mode == 'tfidf':
            self.sklearn_tfidf_embed = self._fit_sklearn_tfidf()
        elif mode == 'my_tfidf':
            self.our_tfidf_embed = self._compute_our_tfidf()
        elif mode == 'bm25':
            self.bm25_embed = self._compute_bm25()
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose from 'tfidf', 'my_tfidf', 'bm25'")
        
    def _build_vocab(self):
        vocab = set()
        for doc in self.tokenized_corpus:
            vocab.update(doc)
        return {word: idx for idx, word in enumerate(sorted(vocab))}
    
    def _compute_doc_freqs(self):
        doc_freqs = Counter()
        for doc in self.tokenized_corpus:
            doc_freqs.update(set(doc))
        return doc_freqs
    
    def _fit_sklearn_tfidf(self):
        self.vectorizer = TfidfVectorizer(tokenizer=self.tokenizer, ngram_range=self.ngram_range, max_features=self.max_features)
        return self.vectorizer.fit_transform(self.corpus)

    def _compute_our_tfidf(self):
        num_docs = len(self.tokenized_corpus)
        idf = {word: math.log(num_docs / (freq + 1)) + 1 for word, freq in self.doc_freqs.items()}
        
        embed = []
        for doc in self.tokenized_corpus:
            doc_tf = Counter(doc)
            doc_embed = [0] * len(self.vocab)
            for word, freq in doc_tf.items():
                if word in self.vocab:
                    idx = self.vocab[word]
                    doc_embed[idx] = (freq / len(doc)) * idf[word]
            embed.append(doc_embed)
        
        return csr_matrix(embed)
    
    def _compute_bm25(self, k1=1.5, b=0.75):
        num_docs = len(self.tokenized_corpus)
        idf = {word: math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1) for word, freq in self.doc_freqs.items()}
        
        embed = []
        for doc in self.tokenized_corpus:
            doc_tf = Counter(doc)
            doc_embed = [0] * len(self.vocab)
            doc_len = len(doc)
            for word, freq in doc_tf.items():
                if word in self.vocab:
                    idx = self.vocab[word]
                    numerator = idf[word] * freq * (k1 + 1)
                    denominator = freq + k1 * (1 - b + b * doc_len / self.avgdl)
                    doc_embed[idx] = numerator / denominator
            embed.append(doc_embed)
        
        return csr_matrix(embed)
    
    def get_embedding(self):
        if self.mode == 'tfidf':
            return self.sklearn_tfidf_embed
        elif self.mode == 'my_tfidf':
            return self.our_tfidf_embed
        elif self.mode == 'bm25':
            return self.bm25_embed
    
    def transform(self, query: str) -> np.ndarray:
        if self.mode == 'tfidf':
            return self.transform_sklearn_tfidf(query)
        elif self.mode == 'my_tfidf':
            return self.transform_our_tfidf(query)
        elif self.mode == 'bm25':
            return self.transform_bm25(query)
    
    def transform_sklearn_tfidf(self, query: str) -> np.ndarray:
        query_vector = self.vectorizer.transform([query])
        return query_vector
    
    def transform_our_tfidf(self, query: str) -> np.ndarray:
        tokenized_query = self.tokenizer(query)
        query_vector = [0] * len(self.vocab)
        for word in tokenized_query:
            if word in self.vocab:
                idx = self.vocab[word] 
                query_vector[idx] = 1
        return csr_matrix([query_vector])
    
    def transform_bm25(self, query: str) -> np.ndarray:
        return self.transform_our_tfidf(query)
    
    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump({
                'tokenizer': self.tokenizer,
                'ngram_range': self.ngram_range,
                'max_features': self.max_features,
                'mode': self.mode,
                'vocab': self.vocab,
                'idf': getattr(self, 'idf', None),
                'avgdl': getattr(self, 'avgdl', None),
                'vectorizer': getattr(self, 'vectorizer', None)
            }, f)
            
    @classmethod
    def load(cls, filename: str):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(
            tokenizer=data['tokenizer'],
            ngram_range=data['ngram_range'],
            max_features=data['max_features'],
            mode=data['mode']
        )
        instance.vocab = data['vocab']
        instance.idf = data['idf']
        instance.avgdl = data['avgdl']
        instance.vectorizer = data['vectorizer']
        
        return instance