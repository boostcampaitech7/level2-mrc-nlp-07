from datasets import load_from_disk, Dataset
from konlpy.tag import Okt
import pandas as pd
import os

# 데이터셋 로드
dataset_path = "/data/ephemeral/home/level2-mrc-nlp-07/data/train_dataset"  # 실제 데이터셋 경로로 대체
data = load_from_disk(dataset_path)

def preprocess_dataset(data_split):
    okt = Okt()  # 형태소 분석기 초기화
    processed_data = []

    # 데이터셋을 순회하며 각 항목을 처리
    for idx in range(len(data_split)):
        item = data_split[idx]  # 각 항목 접근
        question = item['question']  # 질문 가져오기
        nouns = okt.nouns(question)  # 명사 추출
        unique_nouns = list(set(nouns))  # 중복 제거
        
        # 명사들을 쉼표로 구분하여 추가
        nouns_string = ', '.join(unique_nouns)
        processed_question = f"{nouns_string}, {question}"  # 최종 질문 생성
        
        # 수정된 데이터 포인트 생성
        processed_data.append({
            'document_id': item['document_id'],
            'id': item['id'],
            'title': item['title'],
            'context': item['context'],
            'question': processed_question,
            'answers': item['answers']
        })
    
    return processed_data

# train 데이터셋 전처리
train_data = data['train']  # 'train' 분할 선택
processed_train_dataset = preprocess_dataset(train_data)

# validation 데이터셋 전처리 (필요한 경우)
validation_data = data['validation']  # 'validation' 분할 선택
processed_validation_dataset = preprocess_dataset(validation_data)

# DataFrame으로 변환
processed_train_df = pd.DataFrame(processed_train_dataset)
processed_validation_df = pd.DataFrame(processed_validation_dataset)

# 결과 확인
print("Train Data:")
print(processed_train_df.head())
print("\nValidation Data:")
print(processed_validation_df.head())

# 필요한 경우 전처리된 데이터셋 저장
# 데이터셋 경로 지정
train_json_path = '/data/ephemeral/home/level2-mrc-nlp-07/data/processed_data/train_dataset/train/processed_train_dataset.json'
validation_json_path = '/data/ephemeral/home/level2-mrc-nlp-07/data/processed_data/train_dataset/validation/processed_validation_dataset.json'
train_arrow_path = '/data/ephemeral/home/level2-mrc-nlp-07/data/processed_data/train_dataset/train/processed_train_dataset.arrow'
validation_arrow_path = '/data/ephemeral/home/level2-mrc-nlp-07/data/processed_data/train_dataset/validation/processed_validation_dataset.arrow'

# # 디렉터리 생성
# os.makedirs(os.path.dirname(train_json_path), exist_ok=True)
# os.makedirs(os.path.dirname(validation_json_path), exist_ok=True)

processed_train_df.to_json(train_json_path, orient='records', lines=True)
processed_validation_df.to_json(validation_json_path, orient='records', lines=True)

# pandas DataFrame을 Dataset 객체로 변환
train_dataset = Dataset.from_pandas(processed_train_df)
validation_dataset = Dataset.from_pandas(processed_validation_df)

# Arrow 형식으로 저장
train_dataset.save_to_disk(train_arrow_path)
validation_dataset.save_to_disk(validation_arrow_path)
