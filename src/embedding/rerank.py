import numpy as np
import ast
import random
import torch
import pandas as pd
from transformers import (BertModel,
                          BertPreTrainedModel,
                          AutoTokenizer,)
from typing import List, NoReturn, Optional
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from score.ranking import check_original_in_context, calculate_reverse_rank_score, calculate_linear_score
from utils.timer import timer

torch.manual_seed(2023)
torch.cuda.manual_seed(2023)
np.random.seed(2023)
random.seed(2023)
    
        
class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids,
                attention_mask=None, token_type_ids=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

        pooled_output = outputs[1] #CLS 토큰에 해당하는 임베딩
        return pooled_output

class DenseRerank:
    def __init__(self, model_name_or_path="klue/bert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.q_encoder = BertEncoder.from_pretrained('./bert_encoders/q_encoder')
        self.p_encoder = BertEncoder.from_pretrained('./bert_encoders/p_encoder')

        if torch.cuda.is_available():
            self.q_encoder.cuda()
            self.p_encoder.cuda()
            print("GPU enabled")

        self.q_encoder.eval()
        self.p_encoder.eval()

    def embed(self, texts: List[str], encoder) -> torch.Tensor:
        """
        Note: 텍스트 리스트를 BERT 임베딩으로 변환
        """
        inputs = self.tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt").to('cuda')
        with torch.no_grad():
            embeddings = encoder(**inputs)
        return embeddings.cpu()  # (num_texts, hidden_dim)

    def rerank(self, df: pd.DataFrame, topk: int = 20) -> pd.DataFrame:
        """
        주어진 DataFrame 내 'retrieval_context' 열의 후보 패시지들을 유사도 점수로 재정렬합니다.
        """
        new_ranking = []

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            query = row["question"]
            passages = row["retrieval_context"]

            query_emb = self.embed([query], self.q_encoder)  # (1, hidden_dim)
            passage_embs = self.embed(passages, self.p_encoder)  # (20, hidden_dim)

            scores = torch.matmul(query_emb, passage_embs.T).squeeze(0)  # (20,)
            ranked_indices = torch.argsort(scores, descending=True)[:topk]

            new_ranking.append([passages[i] for i in ranked_indices])

        df["retrieval_context"] = new_ranking
        return df
    
    
if __name__ == "__main__":
    reranker = DenseRerank(model_name_or_path="klue/bert-base")
    with timer("read csv"):
        df = pd.read_csv("top_k20.csv")
    with timer("passage setting"):    
        df['retrieval_context'] = df['retrieval_context'].apply(ast.literal_eval)
    
    with timer("reranking passages"):
        reranked_df = reranker.rerank(df, topk=20) 
    
    reranked_df["correct"] = reranked_df.apply(check_original_in_context, axis=1)
    reranked_df["rmm_score"] = reranked_df.apply(calculate_reverse_rank_score, axis=1)
    reranked_df["linear_score"] = reranked_df.apply(calculate_linear_score, axis=1)
    print(
        "correct retrieval result",
        reranked_df["correct"].sum() / len(reranked_df),
    )
    print(
        "rr retrieval result",
        reranked_df["rmm_score"].sum() / len(reranked_df)
    )
    print(
        "linear retrieval result",
        reranked_df["linear_score"].sum() / len(reranked_df)
    )
   
    column_names = reranked_df.columns.tolist()
    print(column_names)
    reranked_df.to_csv('rerank.csv',index=False)