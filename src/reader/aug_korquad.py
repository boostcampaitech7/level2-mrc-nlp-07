import os
import pandas as pd
from datasets import load_from_disk, load_dataset, Dataset

# 기존 데이터셋 로드
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(base_dir, '..', 'data', 'train_dataset')
existing_data = load_from_disk(dataset_path)

# korquad 데이터셋 로드
squad_dataset_path = "squad_kor_v1"
train_dataset = load_dataset(squad_dataset_path, split="train")  # SQuAD-Kor v1 훈련 데이터 로드
val_dataset = load_dataset(squad_dataset_path, split="validation")  # SQuAD-Kor v1 검증 데이터 로드

# SQuAD 데이터를 기존 형식으로 변환하는 함수
def transform_squad_format(data):
    transformed_data = []
    for example in data:
        title = example.get('title', "SQuAD-Kor v1")  # title이 없을 경우 기본값 사용
        for i in range(len(example['answers']['text'])):  # 여러 정답 처리
            transformed_data.append({
                'id': example['id'],
                'title': title,
                'context': example['context'],
                'question': example['question'],
                'answers': {
                    'answer_start': [example['answers']['answer_start'][i]],  # 정답 시작 위치 리스트
                    'text': [example['answers']['text'][i]]  # 정답 텍스트 리스트
                },
                'document_id': 0,  # 기본값 0 설정, 이후에 실제 ID로 변경 가능
            })
    return transformed_data

# 기존 데이터셋을 리스트로 변환
processed_train_existing_data = [example for example in existing_data['train']]  # Dataset -> list
processed_val_existing_data = [example for example in existing_data['validation']]  # Dataset -> list

# SQuAD-Kor v1 데이터셋 변환
processed_squad_train = transform_squad_format(train_dataset)
processed_squad_val = transform_squad_format(val_dataset)

# 데이터 병합 (훈련 데이터)
final_train_dataset = processed_train_existing_data + processed_squad_train

# 검증 데이터 병합 (기존 데이터와 SQuAD-Kor v1)
final_validation_dataset = processed_val_existing_data + processed_squad_val

# DataFrame으로 변환
train_df = pd.DataFrame(final_train_dataset)
validation_df = pd.DataFrame(final_validation_dataset)

# 결과 확인
print("Train Data:")
print(train_df.head())
print("\nValidation Data:")
print(validation_df.head())

# 데이터셋 경로 지정
train_arrow_path = os.path.join(base_dir, '..', 'data', 'noun_dataset', 'train')
validation_arrow_path = os.path.join(base_dir, '..', 'data', 'noun_dataset', 'validation')

# 디렉터리 생성
os.makedirs(os.path.dirname(train_arrow_path), exist_ok=True)
os.makedirs(os.path.dirname(validation_arrow_path), exist_ok=True)

# pandas DataFrame을 Dataset 객체로 변환
train_dataset_final = Dataset.from_pandas(train_df)
validation_dataset_final = Dataset.from_pandas(validation_df)

# Arrow 형식으로 저장
train_dataset_final.save_to_disk(train_arrow_path)
validation_dataset_final.save_to_disk(validation_arrow_path)

print("Datasets saved successfully.")
