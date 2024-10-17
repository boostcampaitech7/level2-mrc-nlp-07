import math
from collections import Counter
from scipy.sparse import csr_matrix


class Tfidf:
    def __init__(self, vocab, doc_freqs, tokenized_docs, ngram_range, max_features):
        self.vocab = vocab
        self.doc_freqs = doc_freqs
        self.tokenized_docs = tokenized_docs
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.embedding_matrix = None
    
    def fit(self):
        self.embedding_matrix = self._fit_transform()
    
    def get_embedding(self):
        if self.embedding_matrix is None:
            self.fit()
        return self.embedding_matrix
        
    def _fit_transform(self):
        num_docs = len(self.tokenized_docs)
        idf = {word: math.log(num_docs / (freq + 1)) + 1 for word, freq in self.doc_freqs.items()}
        embed = []
        for doc in self.tokenized_docs:
            doc_tf = Counter(doc)
            doc_embed = [0] * len(self.vocab)
            for word, freq in doc_tf.items():
                if word in self.vocab:
                    idx = self.vocab[word]
                    doc_embed[idx] = (freq / len(doc)) * idf[word]
            embed.append(doc_embed)
        
        print('Finish TF-IDF Embedding')
        return csr_matrix(embed)
    
    def transform(self, tokenized_query: str):
        query_vector = [0] * len(self.vocab)
        for word in tokenized_query:
            if word in self.vocab:
                idx = self.vocab[word] 
                query_vector[idx] = 1
        return csr_matrix([query_vector])