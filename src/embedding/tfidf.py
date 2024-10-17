import math
from collections import Counter
from scipy.sparse import csr_matrix #   

class Tfidf:
    def __init__(self, vocab, doc_freqs, docs, tokenizer, ngram_range, max_features):
        self.vocab = vocab
        self.doc_freqs  = doc_freqs
        self.tokenizer = tokenizer
        self.tokenized_docs = [tokenizer(doc) for doc in docs]
    
    def embedding(self):
        return self._compute_our_tfidf()
        
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
        
        return csr_matrix(embed) #
    
    def transform(self, query: str):
        tokenized_query = self.tokenizer(query)
        query_vector = [0] * len(self.vocab)
        for word in tokenized_query:
            if word in self.vocab:
                idx = self.vocab[word] 
                query_vector[idx] = 1
        return csr_matrix([query_vector])