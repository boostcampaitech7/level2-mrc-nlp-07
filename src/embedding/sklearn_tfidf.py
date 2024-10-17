from sklearn.feature_extraction.text import TfidfVectorizer

class SklearnTfidf:
    def __init__(self, docs, tokenizer, ngram_range, max_features):
        self.docs = docs
        self.vectorizer = TfidfVectorizer(tokenizer=tokenizer, ngram_range=ngram_range, max_features=max_features)
        
    
    def embedding(self):
        return self.vectorizer.fit_transform(self.docs)
    
    def transform(self, query):
        return self.vectorizer.transform([query])