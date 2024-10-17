from collections import Counter
from typing import List
import numpy as np
import src.embedding as embedding_function
import pickle


class SparseEmbedding:
    def __init__(self, docs: List[str], tokenizer=None, ngram_range=(1,2), max_features:int=50000, mode: str = 'tfidf'):
        self.docs = docs
        self.tokenizer = tokenizer if tokenizer else lambda x: x.split()
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.mode = mode
        self.tokenized_docs = [tokenizer(doc) for doc in docs]
        self.embedding_function = None
        
    def get_embedding_function(self):
        if self.mode == 'tfidf':
            self.embedding_function =  embedding_function.SklearnTfidf(
                tokenized_docs=self.docs,
                tokenizer=self.tokenizer,
                ngram_range=self.ngram_range,
                max_features=self.max_features
            )
            
        else:
            vocab = self._build_vocab()
            doc_freqs = self._compute_doc_freqs()
            if self.mode == 'my_tfidf':
                self.embedding_function = embedding_function.Tfidf(
                    vocab=vocab,
                    doc_freqs=doc_freqs,
                    docs=self.docs,
                    tokenizer=self.tokenizer,
                    ngram_range=self.ngram_range,
                    max_features=self.max_features
                )
                
            elif self.mode == 'bm25':
                self.embedding_function = embedding_function.Bm25(
                    vocab=vocab,
                    doc_freqs=doc_freqs,
                    docs=self.docs,
                    tokenizer=self.tokenizer,
                    ngram_range=self.ngram_range,
                    max_features=self.max_features
                )
        
    def embedding(self):
        return self.embedding_function.embedding
    
    def transform(self, query: str) -> np.ndarray:
        tokenized_query = self.tokenizer(query)
        if self.mode == 'tfidf':
            return self.embedding_function.transform(query)
        else:
            return self.embedding_function.transform(tokenized_query)
        
    def _build_vocab(self):
        vocab = set()
        for doc in self.tokenized_docs:
            vocab.update(doc)
        return {word: idx for idx, word in enumerate(sorted(vocab))}
    
    def _compute_doc_freqs(self):
        doc_freqs = Counter()
        for doc in self.tokenized_docs:
            doc_freqs.update(set(doc))
        return doc_freqs
    
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