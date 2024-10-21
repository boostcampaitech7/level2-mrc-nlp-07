from collections import Counter
from typing import List
import numpy as np
import src.retriever.embedding as embedding_function
import pickle
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer


class SparseEmbedding:
    def __init__(self, docs: List[str], tokenizer=None, ngram_range:tuple=(1,2), max_features:int=50000, mode: str = 'tfidf'):
        self.docs = docs
        self.tokenizer = tokenizer if tokenizer else lambda x: x.split()
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.mode = mode
        self.embedding_function = None
        self.tokenized_docs = None
        self.vocab = None
        self.doc_freqs = None
        if docs is not None:
            print('Start Initializing...')
            self.tokenized_docs = [self.tokenizer(doc) for doc in tqdm(docs, desc="Tokenizing...")]
            print('Generating n-grams and building vocabulary...')
            self._generate_ngrams_and_update_vocab()
            self.initialize_embedding_function()
        
    def initialize_embedding_function(self):
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
                vocab = self.vocab,
                doc_freqs = self.doc_freqs,
                tokenized_docs = self.tokenized_docs,
                ngram_range = self.ngram_range,
                max_features = self.max_features
            )
            
        elif self.mode == 'bm25':
            self.embedding_function = embedding_function.Bm25(
                vocab = self.vocab,
                doc_freqs = self.doc_freqs,
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
    
    def _generate_ngrams_and_update_vocab(self):
        new_vocab = Counter()
        new_doc_freqs = Counter()
        
        for doc in tqdm(self.tokenized_docs, desc="Generating n-grams"):
            doc_ngrams = self._get_ngrams(doc)
            new_vocab.update(doc_ngrams)
            new_doc_freqs.update(set(doc_ngrams))
        
        if self.max_features:
            new_vocab = dict(new_vocab.most_common(self.max_features))
        
        
        self.vocab = {word: idx for idx, word in enumerate(new_vocab)}
        self.doc_freqs = {word: new_doc_freqs[word] for word in self.vocab}
        
    def _get_ngrams(self, tokens):
        n_grams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            n_grams.extend([' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])
        return n_grams
    
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