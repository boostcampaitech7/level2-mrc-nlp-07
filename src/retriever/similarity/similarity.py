import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class ComputeSimilarity:
    @staticmethod
    def compute_jaccard_similarity(query_vecs, p_embedding):
        """
        Compute Jaccard similarity between query vectors and passage embeddings.
        
        :param query_vecs: sparse matrix of query vectors
        :param p_embedding: sparse matrix of passage embeddings
        :return: matrix of Jaccard similarities
        """
        query_binary = query_vecs.astype(bool).astype(int)
        p_binary = p_embedding.astype(bool).astype(int)

        intersection = query_binary @ p_binary.T
        query_sum = query_binary.sum(axis=1)
        p_sum = p_binary.sum(axis=1)
        union = query_sum[:, np.newaxis] + p_sum - intersection

        return intersection / union

    @staticmethod
    def compute_levenshtein_distance(queries, passages):
        """
        Compute Levenshtein distance between queries and passages.
        
        :param queries: list of query strings
        :param passages: list of passage strings
        :return: matrix of Levenshtein distances
        """
        vectorizer = CountVectorizer(binary=True)
        query_tokens = vectorizer.fit_transform(queries).toarray()
        passage_tokens = vectorizer.transform(passages).toarray()

        distances = np.zeros((len(queries), len(passages)))
        for i, query in enumerate(query_tokens):
            for j, passage in enumerate(passage_tokens):
                distances[i, j] = -ComputeSimilarity._levenshtein_distance(query, passage)

        return distances

    @staticmethod
    def _levenshtein_distance(s1, s2):
        """
        Compute Levenshtein distance between two binary arrays.
        
        :param s1: first binary array
        :param s2: second binary array
        :return: Levenshtein distance
        """
        len_s1, len_s2 = len(s1), len(s2)
        dp = np.zeros((len_s1 + 1, len_s2 + 1), dtype=int)
        dp[0, :] = np.arange(len_s2 + 1)
        dp[:, 0] = np.arange(len_s1 + 1)

        for i in range(1, len_s1 + 1):
            for j in range(1, len_s2 + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i, j] = dp[i - 1, j - 1]
                else:
                    dp[i, j] = 1 + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

        return dp[len_s1, len_s2]