import torch 
from transformers import BartForConditionalGeneration, BartTokenizer
from datasets import load_from_disk, Dataset
import os
from tqdm import tqdm
import pandas as pd

# 1. Arrow 파일 로드
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(base_dir, '..', 'data', 'train_dataset')
data = load_from_disk(dataset_path)

# 2. BART 모델과 토크나이저 로드 (Question Generation 모델)
model_name = "facebook/bart-base"  # 적절한 모델로 변경 가능
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# 3. 질문 생성 함수
def generate_question(context, num_questions=1):
    input_ids = tokenizer.encode(context, return_tensors="pt", truncation=True, max_length=512)

    # 질문 생성
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=50, num_return_sequences=num_questions, num_beams=5, early_stopping=True)

    # 생성된 질문 디코딩
    questions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return questions[0] if questions else None

# 4. 데이터셋을 순회하며 기존 질문과 생성된 질문을 각각 별도 데이터로 추가
def preprocess_dataset(data_split):
    processed_data = []
    
    # tqdm으로 진행률 출력
    for item in tqdm(data_split, desc="Processing data"):
        context = item['context']
        
        # 기존 질문을 추가 (기존 데이터)
        processed_data.append({
            'document_id': item['document_id'],
            'id': item['id'],
            'title': item['title'],
            'context': item['context'],
            'question': item['question'],  # 기존 질문
            'answers': item['answers']
        })
        
        # 새로운 질문 생성 및 추가 (새로운 데이터)
        generated_question = generate_question(context)
        if generated_question:  # 질문이 생성된 경우만 추가
            processed_data.append({
                'document_id': item['document_id'],
                'id': f"{item['id']}-gen",  # 새로운 ID 생성
                'title': item['title'],
                'context': item['context'],
                'question': generated_question,  # 생성된 질문
                'answers': item['answers']  # 답변은 동일
            })
    
    return processed_data

# 5. train 데이터셋 전처리 및 질문 생성
train_data = data['train']
processed_train_data = preprocess_dataset(train_data)

# 6. validation 데이터셋 전처리 및 질문 생성
validation_data = data['validation']
processed_validation_data = preprocess_dataset(validation_data)

# 7. DataFrame으로 변환 후 Arrow 형식으로 저장
train_df = pd.DataFrame(processed_train_data)
validation_df = pd.DataFrame(processed_validation_data)

train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(validation_df)

# Arrow 형식으로 저장할 경로 설정 (상대 경로)
processed_data_path = os.path.join(base_dir, '..', 'data', 'generated_questions_dataset')
train_arrow_path = os.path.join(processed_data_path, "train")
validation_arrow_path = os.path.join(processed_data_path, "validation")

# 디렉토리 생성
os.makedirs(train_arrow_path, exist_ok=True)
os.makedirs(validation_arrow_path, exist_ok=True)

# 저장
train_dataset.save_to_disk(train_arrow_path)
validation_dataset.save_to_disk(validation_arrow_path)

print(f"Processed datasets with separate original and generated questions saved to: {processed_data_path}")
