from sklearn.feature_extraction.text import TfidfVectorizer


class SklearnTfidf:
    def __init__(self, docs, tokenizer, ngram_range, max_features):
        self.docs = docs
        self.vectorizer = TfidfVectorizer(tokenizer=tokenizer, ngram_range=ngram_range, max_features=max_features)
        self.embedding_matrix = None
    
    def fit(self):
        self.embedding_matrix = self.vectorizer.fit_transform(self.docs)
        print('Finish Sklearn TF-IDF Embedding')
    
    def get_embedding(self):
        if self.embedding_matrix is None:
            self.fit()
        return self.embedding_matrix
    
    def transform(self, query):
        if isinstance(query, str):
            return self.vectorizer.transform([query])
        else:
            return self.vectorizer.transform(query)