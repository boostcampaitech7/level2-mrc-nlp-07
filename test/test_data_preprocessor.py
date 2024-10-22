import pytest
from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast, EvalPrediction
from src.reader.data_controller.data_processor import DataPreProcessor, DataPostProcessor
from src.utils.arguments import DataTrainingArguments
from transformers import BertTokenizerFast

@pytest.fixture
def data_args():
    return DataTrainingArguments(dataset_name="/data/ephemeral/home/level2-mrc-nlp-07/data/train_dataset/train")  # 절대 경로로 수정

@pytest.fixture
def sample_data(data_args):  # data_args를 인자로 추가
    # sample_data는 이제 DataTrainingArguments에서 지정된 경로에서 데이터를 로드합니다.
    dataset = load_from_disk(data_args.dataset_name)
    return dataset

@pytest.fixture
def tokenizer():
    # 샘플 토크나이저 생성
    return BertTokenizerFast.from_pretrained('bert-base-uncased')

def test_data_preprocessor(sample_data, tokenizer, data_args):
    processor = DataPreProcessor()
    
    # 데이터 전처리
    processed_dataset = processor.process('train', tokenizer, data_args, sample_data)
    
    # 결과 검증
    assert 'input_ids' in processed_dataset[0]
    assert 'start_positions' in processed_dataset[0]
    assert 'end_positions' in processed_dataset[0]
