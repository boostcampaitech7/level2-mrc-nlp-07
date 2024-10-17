import math
from collections import Counter
from scipy.sparse import csr_matrix

class Bm25:
    def __init__(self, vocab, doc_freqs, tokenized_docs, k1=1.5, b=0.75):
        self.vocab = vocab
        self.doc_freqs = doc_freqs
        self.tokenized_docs = tokenized_docs
        
        self.k1 = k1
        self.b = b
        self.average_document_length = sum(len(doc) for doc in self.tokenized_docs) / len(self.tokenized_docs)
    
    def embedding(self):
        return self._fit_transform
        
    def _fit_transform(self):
        num_docs = len(self.tokenized_docs)
        idf = {word: math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1) for word, freq in self.doc_freqs.items()}
        
        embed = []
        for doc in self.tokenized_docs:
            doc_tf = Counter(doc)
            doc_embed = [0] * len(self.vocab)
            doc_len = len(doc)
            for word, freq in doc_tf.items():
                if word in self.vocab:
                    idx = self.vocab[word]
                    numerator = idf[word] * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.average_document_length)
                    doc_embed[idx] = numerator / denominator
            embed.append(doc_embed)
        
        return csr_matrix(embed)
    
    def transform(self, query: str):
        query_vector = [0] * len(self.vocab)
        for word in query:
            if word in self.vocab:
                idx = self.vocab[word] 
                query_vector[idx] = 1
        return csr_matrix([query_vector])