from collections import Counter
from typing import List
import numpy as np
import src.embedding as embedding_function
import pickle
from tqdm import tqdm


class SparseEmbedding:
    def __init__(self, docs: List[str], tokenizer=None, ngram_range:tuple=(1,2), max_features:int=50000, mode: str = 'tfidf'):
        self.docs = docs
        self.tokenizer = tokenizer if tokenizer else lambda x: x.split()
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.mode = mode
        self.embedding_function = None
        self.tokenized_docs = None
        
        if docs is not None:
            print('Start Initializing...')
            self.tokenized_docs = [self.tokenizer(doc) for doc in tqdm(docs, desc="Tokenizing...")]
            self.initialize_embedding_function()
        
    def initialize_embedding_function(self):
        print('Building Vocab')
        vocab = self._build_vocab()
        print('Calculate doc frequency')
        doc_freqs = self._compute_doc_freqs()
        print(f'Current mode : {self.mode}')
        if self.mode == 'tfidf':
            self.embedding_function = embedding_function.SklearnTfidf(
                docs=self.docs,
                tokenizer=self.tokenizer,
                ngram_range=self.ngram_range,
                max_features=self.max_features
            )
            
        elif self.mode == 'my_tfidf':
            self.embedding_function = embedding_function.Tfidf(
                vocab = vocab,
                doc_freqs = doc_freqs,
                tokenized_docs = self.tokenized_docs,
                ngram_range = self.ngram_range,
                max_features = self.max_features
            )
            
        elif self.mode == 'bm25':
            self.embedding_function = embedding_function.Bm25(
                vocab = vocab,
                doc_freqs = doc_freqs,
                tokenized_docs = self.tokenized_docs,
                ngram_range = self.ngram_range,
                max_features = self.max_features
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Choose from 'tfidf', 'my_tfidf', 'bm25'")
        
        # Fit the embedding function if it has a fit method
        print('End Initialization')
        if hasattr(self.embedding_function, 'fit'):
            self.embedding_function.fit()
    
    def get_embedding(self):
        if self.embedding_function is None:
            raise ValueError("Embedding function has not been initialized. Call initialize_embedding_function first.")
        return self.embedding_function.get_embedding()
    
    def transform(self, query: str) -> np.ndarray:
        tokenized_query = self.tokenizer(query)
        if self.mode == 'tfidf':
            return self.embedding_function.transform(query)
        else:
            return self.embedding_function.transform(tokenized_query)
        
    def _build_vocab(self):
        vocab = set()
        for doc in tqdm(self.tokenized_docs, desc="빌딩 어휘"):
            vocab.update(doc)
        return {word: idx for idx, word in enumerate(sorted(vocab))}
    
    def _compute_doc_freqs(self):
        doc_freqs = Counter()
        for doc in tqdm(self.tokenized_docs, desc="문서 빈도 계산"):
            doc_freqs.update(set(doc))
        return doc_freqs
    
    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump({
                'docs': self.docs,
                'tokenizer': self.tokenizer,
                'ngram_range': self.ngram_range,
                'max_features': self.max_features,
                'mode': self.mode,
                'embedding_function': self.embedding_function
            }, f)
            
    @classmethod
    def load(cls, filename: str):
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
        except (EOFError, pickle.UnpicklingError) as e:
            raise ValueError(f"Error loading SparseEmbedding from {filename}: {str(e)}")

        instance = cls(docs=None)  # Create an empty instance
        instance.__dict__.update(data)  # Update instance attributes with loaded data
        
        # Verify that essential attributes are present
        essential_attrs = ['tokenizer', 'ngram_range', 'max_features', 'mode', 'embedding_function']
        for attr in essential_attrs:
            if not hasattr(instance, attr):
                raise ValueError(f"Loaded data is missing essential attribute: {attr}")

        return instance