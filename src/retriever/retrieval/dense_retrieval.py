import numpy as np
import argparse
import random
import torch
import json
import torch.nn.functional as F
import pandas as pd
from transformers import (BertModel,
                          BertPreTrainedModel,
                          AdamW,
                          TrainingArguments,
                          AutoTokenizer,
                          get_linear_schedule_with_warmup)
from datasets import (Dataset, 
                      concatenate_datasets, 
                      load_from_disk)
from typing import List, NoReturn, Optional
from torch.utils.data import(DataLoader,TensorDataset)
from tqdm import tqdm, trange 
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
  
class DenseRetrieval:
    def __init__(self,
                 data_path: Optional[str] = "./data/train_dataset",
                 model_name_or_path: Optional[str] = "bert-base-multilingual-cased",
                 num_neg: int = 3,
                 use_in_batch_negative_sampling: bool = True,
                 train_args: TrainingArguments = None
                )-> NoReturn:
        self.num_neg = num_neg   
        self.model_name_or_path = model_name_or_path
        self.use_in_batch_negative_sampling = use_in_batch_negative_sampling 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.train_args = train_args
        self.full_ds = self._load_full_data(data_path)
        self.corpus = self._make_corpus()
        self.p_with_neg = self._make_p_with_neg_before_batch()
        self.q_seqs, self.p_seqs = self._set_q_and_s_seqs()        
        self.train_dataset = TensorDataset(self.p_seqs['input_ids'], self.p_seqs['attention_mask'], self.p_seqs['token_type_ids'],
                                           self.q_seqs['input_ids'], self.q_seqs['attention_mask'], self.q_seqs['token_type_ids'])        
        
        self.p_encoder = BertEncoder.from_pretrained(self.model_name_or_path)
        self.q_encoder = BertEncoder.from_pretrained(self.model_name_or_path)
        self.docs = self._initialize_from_wiki(context_path="/data/ephemeral/home/level2-mrc-nlp-07/data/wikipedia_documents.json")

    def _initialize_from_wiki(self, context_path: str):
        with open(context_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)

        docs = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        print(f"Lengths of unique contexts : {len(docs)}")
        return docs
        
    def _load_full_data(
        self,
        data_path: Optional[str] = "./data/train_dataset"
        ) -> Dataset:
        """
        Arguments:
            data_path:
                데이터가 보관되어 있는 경로입니다.
            
        Returns:
            data_path 내 train 데이터와 validation 데이터를 합산한 통합 데이터를 반환합니다.
        """
        org_dataset = load_from_disk(data_path)
        full_ds = org_dataset
        print("*" * 40, "query dataset", "*" * 40)
        print(org_dataset)
        full_ds = concatenate_datasets(
            [
                org_dataset["train"].flatten_indices(),
                org_dataset["validation"].flatten_indices()
            ]
        )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트    
        return full_ds  

    def _make_corpus(self) -> List:
        """
        Arguments:
            full_ds:
                중복없는 corpus를 생성할 dataset 파일입니다.

        Returns:
            중복없는 full_ds로 구성된 순차적 corpus list를 반환합니다. 
        """
        corpus = list(set([example['context'] for example in self.full_ds]))
        return corpus
    
    def _make_p_with_neg_before_batch(self) -> List:
        """
        Returns:
            1개의 context 당 num_neg개의 negative sample이 이어진 p_with_neg List를 반환합니다.
        
        Note:
            p_with_neg의 길이는 (positive context의 개수) * (1+num_neg)가 됩니다.
        """
        corpus_list = np.array(self.corpus) # list -> numpy
        p_with_neg = []

        for c in self.full_ds['context']: # training_dataset의 모든 context에 대해서 
            while True:
                neg_idxs = np.random.randint(len(corpus_list), size=self.num_neg) # num_neg만큼 랜덤 추출

                if not c in corpus_list[neg_idxs]: # c가 courpus[neg_index]랑 겹치지 않으면 (neg sample이면)
                    p_neg = corpus_list[neg_idxs]

                    p_with_neg.append(c)
                    p_with_neg.extend(p_neg)
                    break 
        return p_with_neg        
       
    def _set_s_seqs_with_neg(self):
        """
        Returns:
            p_seqs의 tokenized 결과를 (총 context 개수)*(1+num_neg)*(max_length)로 변경한 p_seqs
        """
        with timer("tokenizing p_seq"):
            p_seqs = self.tokenizer(self.p_with_neg, padding="max_length", truncation=True, return_tensors='pt')
        
        max_len = p_seqs['input_ids'].size(-1) #맨마지막 dim
        p_seqs['input_ids'] = p_seqs['input_ids'].view(-1, self.num_neg+1, max_len)
        p_seqs['attention_mask'] = p_seqs['attention_mask'].view(-1, self.num_neg+1, max_len)
        p_seqs['token_type_ids'] = p_seqs['token_type_ids'].view(-1, self.num_neg+1, max_len)
        
        print("p_seqs['input_ids']:",p_seqs['input_ids'].size()) #(num_example, pos + neg, max_len)
        print("p_seqs['attention_mask']:",p_seqs['attention_mask'].size()) #(num_example, pos + neg, max_len)
        print("p_seqs['token_type_ids']:",p_seqs['token_type_ids'].size()) #(num_example, pos + neg, max_len)
        return p_seqs
   
    def _set_q_and_s_seqs(self):
        """
        Returns:
            in batch 여부에 따른 q_seqs와 p_seqs 설정
        Notes:
            q_seqs의 경우 in batch 여부와 관계 없이 동일
            p_seqs의 경우 in batch가 True일 때 train 내부에서 negative sampling 진행,
            in batch가 False일 때 미리 제작해둔 negative sampling 적용
        """
        q_seqs = self.tokenizer(self.full_ds['question'], padding="max_length", truncation = True, return_tensors = "pt")
        if self.use_in_batch_negative_sampling:
            p_seqs = self.tokenizer(self.full_ds['context'], padding="max_length", truncation = True, return_tensors = "pt")
        else:
            p_seqs = self._set_s_seqs_with_neg()
        return q_seqs, p_seqs
            
    def _train_encoders_with_in_batch_negative_sampling(self, args):
        import torch.nn as nn

        batch_size = args.per_device_train_batch_size
        train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size)

        optimizer = AdamW([
            {"params": self.p_encoder.parameters(), "weight_decay": args.weight_decay},
            {"params": self.q_encoder.parameters(), "weight_decay": args.weight_decay},
        ], lr=args.learning_rate, eps=args.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=len(train_dataloader) * args.num_train_epochs
        )

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        loss_fn = nn.CrossEntropyLoss()

        train_iterator = trange(args.num_train_epochs, desc="Epoch")
        for _ in train_iterator:
            with tqdm(train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:
                    self.p_encoder.train()
                    self.q_encoder.train()

                    # 질문 및 지문 임베딩 계산
                    q_inputs = {key: batch[i].to(args.device) for i, key in enumerate(["input_ids", "attention_mask", "token_type_ids"])}
                    p_inputs = {key: batch[i + 3].to(args.device) for i, key in enumerate(["input_ids", "attention_mask", "token_type_ids"])}

                    q_outputs = self.q_encoder(**q_inputs)  # (B, d)
                    p_outputs = self.p_encoder(**p_inputs)  # (B, d)

                    # 유사도 행렬 계산 (B x B)
                    sim_scores = torch.matmul(q_outputs, p_outputs.T)

                    # 손실 계산 (i, i가 정답인 경우)
                    targets = torch.arange(batch_size).to(args.device)  # [0, 1, 2, ..., B-1]
                    loss = loss_fn(sim_scores, targets)
                    tepoch.set_postfix(loss=f"{loss.item():.4f}")

                    # 역전파 및 최적화
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.q_encoder.zero_grad()
                    self.p_encoder.zero_grad()

        return self.p_encoder, self.q_encoder
                                
    def _train_encoders(self, args):
        
        if torch.cuda.is_available():
            self.p_encoder.cuda()
            self.q_encoder.cuda()
            print("GPU enabled")
            
        num_neg = self.num_neg
        dataset = self.train_dataset
        
        batch_size = args.per_device_train_batch_size

        # Dataloader
        train_dataloader = DataLoader(dataset, batch_size=batch_size)

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            {"params": [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon
        )
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=t_total
        )

        # Start training!
        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
        for _ in train_iterator:

            # epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            with tqdm(train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:

                    self.p_encoder.train()
                    self.q_encoder.train()

                    targets = torch.zeros(batch_size).long() # positive example은 전부 첫 번째에 위치하므로
                    targets = targets.to(args.device)

                    p_inputs = {
                        "input_ids": batch[0].view(batch_size * (num_neg + 1), -1).to(args.device),
                        "attention_mask": batch[1].view(batch_size * (num_neg + 1), -1).to(args.device),
                        "token_type_ids": batch[2].view(batch_size * (num_neg + 1), -1).to(args.device)
                    }

                    q_inputs = {
                        "input_ids": batch[3].to(args.device),
                        "attention_mask": batch[4].to(args.device),
                        "token_type_ids": batch[5].to(args.device)
                    }

                    del batch
                    torch.cuda.empty_cache()
                    # (batch_size * (num_neg + 1), emb_dim)
                    p_outputs = self.p_encoder(**p_inputs)
                    # (batch_size, emb_dim)
                    q_outputs = self.q_encoder(**q_inputs)

                    # Calculate similarity score & loss
                    p_outputs = p_outputs.view(batch_size, num_neg + 1, -1)
                    q_outputs = q_outputs.view(batch_size, 1, -1)

                    # (batch_size, num_neg + 1)
                    sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f"{str(loss.item())}")

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.q_encoder.zero_grad()
                    self.p_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()
                    del p_inputs, q_inputs

        return self.p_encoder, self.q_encoder
    
    def get_dense_embedding(self):
        if args.use_in_batch_negative_sampling:
            self.p_encoder, self.q_encoder = self._train_encoders_with_in_batch_negative_sampling(self.train_args)
        else:
            self.p_encoder, self.q_encoder = self._train_encoders(self.train_args)
            
    def retrieve(
        self,
        queries_or_dataset, 
        topk=3
    ):
        """
        Retrieves the top-k passages for a given query or dataset.

        Args:
            model: DenseRetrieval 객체. p_encoder와 q_encoder를 포함.
            queries_or_dataset: 검색할 단일 쿼리(query) 혹은 Dataset.
            valid_corpus: 검색 대상이 될 문서들의 목록.
            topk: 상위 k개의 문서를 반환.

        Returns:
            pd.DataFrame: 쿼리와 검색된 문서를 포함하는 DataFrame.
        """
        valid_corpus = self.docs
        print(len(valid_corpus))
        with timer("tokenizing valid p_seq and load dataloader"):
            valid_seqs = self.tokenizer(valid_corpus, padding="max_length", truncation=True, return_tensors='pt')
            passage_dataset = TensorDataset(
                valid_seqs['input_ids'], valid_seqs['attention_mask'], valid_seqs['token_type_ids']
            )
            passage_dataloader = DataLoader(passage_dataset, batch_size=2)

        with torch.no_grad():
            self.p_encoder.eval()
            self.q_encoder.eval()

            if isinstance(queries_or_dataset, list):  
                queries = queries_or_dataset  # 단일 쿼리 리스트
            elif isinstance(queries_or_dataset, Dataset):
                queries = queries_or_dataset["question"]  # Dataset에서 쿼리 추출
            else:
                raise ValueError("Input must be a list of queries or a Dataset.")

            # Query 임베딩 계산
            q_seqs = self.tokenizer(
                queries, padding="max_length", truncation=True, return_tensors='pt'
            ).to('cuda')
            q_emb = self.q_encoder(**q_seqs).to('cpu')  # (num_queries, emb_dim)
            print(f'q_emb.shape: {q_emb.shape}')
            # Passage 임베딩 계산
            p_embs = []
            for batch in passage_dataloader:

                batch = tuple(t.to('cuda') for t in batch)
                p_inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2]
                }
                p_emb = self.p_encoder(**p_inputs).to('cpu')
                p_embs.append(p_emb)
                
        p_embs = torch.cat(p_embs, dim=0)        
        print(f'p_embs.shape: {p_embs.shape}')
        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
        print(f'dot_prod_scores.shape: {dot_prod_scores.shape}')
        
        doc_scores, doc_indices = torch.topk(dot_prod_scores, topk, dim=1)
        """
            # Passage 임베딩 배열로 변환
            p_embs = torch.Tensor(np.array(p_embs)).squeeze()  # (num_passage, emb_dim)
            print(f'p_embs.shape: {p_embs.shape}')
            # Dot-product score 계산
            dot_prod_scores = q_emb @ p_embs.T  # (num_queries, num_passage)
            print(f'dot_prod_scores.shape: {dot_prod_scores.shape}')
            # 내림차순으로 정렬된 상위 K passage 인덱스와 점수
            doc_scores, doc_indices = torch.topk(dot_prod_scores, topk, dim=1)
        """
            
        # 결과를 DataFrame 형태로 정리
        total = []
        for idx, query in enumerate(queries):
            retrieved_contexts = [valid_corpus[i] for i in doc_indices[idx]]

            tmp = {
                "question": query,
                "retrieval_context": retrieved_contexts,
            }

            # Dataset일 경우 원본 context 및 answers 포함
            if isinstance(queries_or_dataset, Dataset):
                tmp["id"] = queries_or_dataset[idx]["id"]
                if "context" in queries_or_dataset.features:
                    tmp["original_context"] = queries_or_dataset[idx]["context"]
                if "answers" in queries_or_dataset.features:
                    tmp["answers"] = queries_or_dataset[idx]["answers"]

            total.append(tmp)

        return pd.DataFrame(total)
    
    def get_score(self, df):
        df["correct"] = df.apply(check_original_in_context, axis=1)
        df["rr_score"] = df.apply(calculate_reverse_rank_score, axis=1)
        df["linear_score"] = df.apply(calculate_linear_score, axis=1)
        print(
            "correct retrieval",
            df["correct"].sum() / len(df),
        )
        print(
            "reverse rank retrieval",
            df["rr_score"].sum() / len(df)
        )
        print(
            "linear retrieval",
            df["linear_score"].sum() / len(df)
        )
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model_name_or_path",
        metavar="bert-base-multilingual-cased",
        type=str,
        help="",
        default="bert-base-multilingual-cased"
    )
    parser.add_argument("--data_path", 
        metavar="/data/ephemeral/home/level2-mrc-nlp-07/data/train_dataset",
        type=str, 
        help="",
        default="/data/ephemeral/home/level2-mrc-nlp-07/data/train_dataset")
    parser.add_argument("--num_neg", 
        metavar=3, 
        type=int, 
        help="",
        default=3
        )
    parser.add_argument("--use_in_batch_negative_sampling", 
        metavar=False, 
        type=bool, 
        help="",
        default=False
        )
    
    args = parser.parse_args()
    train_args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=2,
        weight_decay=0.01
    )
    denseRetrieval = DenseRetrieval(
        args.data_path,
        args.model_name_or_path,
        args.num_neg,
        args.use_in_batch_negative_sampling,
        train_args
    )
    
    org_dataset = load_from_disk(args.data_path)
    queries = org_dataset['validation']
    
    denseRetrieval.get_dense_embedding()
    with timer("bulk query search"):
        df = denseRetrieval.retrieve(queries,topk=10)
        denseRetrieval.get_score(df)
        
        df.to_csv('output.csv', index=False, encoding='utf-8-sig')
    