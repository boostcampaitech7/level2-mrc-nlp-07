import math
from collections import Counter
from scipy.sparse import csr_matrix, vstack
import numpy as np

class Bm25:
    def __init__(self, vocab, doc_freqs, tokenized_docs, ngram_range, max_features, k1:float=1.5, b:float=0.75):
        self.vocab = vocab
        self.doc_freqs = doc_freqs
        self.tokenized_docs = tokenized_docs
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.k1 = k1
        self.b = b
        self.average_document_length = sum(len(doc) for doc in self.tokenized_docs) / len(self.tokenized_docs)
        self.embedding_matrix = None

    def fit(self):
        self.embedding_matrix = self._fit_transform()

    def get_embedding(self):
        if self.embedding_matrix is None:
            self.fit()
        return self.embedding_matrix

    def _fit_transform(self):
        num_docs = len(self.tokenized_docs)
        idf = {word: math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1) for word, freq in self.doc_freqs.items()}
        
        batch_size = 1000  # 한 번에 처리할 문서 수
        embed_list = []
        
        for i in range(0, num_docs, batch_size):
            batch_docs = self.tokenized_docs[i:i+batch_size]
            batch_embed = []
            
            for doc in batch_docs:
                doc_tf = Counter(doc)
                doc_embed = np.zeros(len(self.vocab), dtype=np.float32)  # float32 사용
                doc_len = len(doc)
                for word, freq in doc_tf.items():
                    if word in self.vocab:
                        idx = self.vocab[word]
                        numerator = idf[word] * freq * (self.k1 + 1)
                        denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.average_document_length)
                        doc_embed[idx] = numerator / denominator
                batch_embed.append(doc_embed)
            
            embed_list.append(csr_matrix(batch_embed))
            print(f'Processed {i+len(batch_docs)}/{num_docs} documents')
        
        print('Finish BM25 Embedding')
        return vstack(embed_list)

    def transform(self, tokenized_query: str):
        query_vector = np.zeros(len(self.vocab), dtype=np.float32)  # float32 사용
        for word in tokenized_query:
            if word in self.vocab:
                idx = self.vocab[word] 
                query_vector[idx] = 1
        return csr_matrix([query_vector])