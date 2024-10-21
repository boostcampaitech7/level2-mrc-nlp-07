from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from datasets import Dataset
from datasets import DatasetDict
from transformers import AutoTokenizer

from src import DataTrainingArguments, DataPostProcessor, DataPreProcessor


@pytest.fixture
def mock_data_args():
    return DataTrainingArguments(
        dataset_name='valid_dataset_path',
        max_seq_length=384,
        pad_to_max_length=False,
        preprocessing_num_workers=1,
        overwrite_cache=False,
        doc_stride=128,
    )


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock(spec=AutoTokenizer)
    tokenizer.padding_side = 'right'
    tokenizer.model_max_length = 512
    tokenizer.return_value = {
        'input_ids': [[101, 102]],
        'overflow_to_sample_mapping': [0],
        'offset_mapping': [[(0, 1), (1, 2)]],
        'start_positions': [0],
        'end_positions': [1],
    }
    return tokenizer


@pytest.fixture
def mock_datasets():
    # Mocking a simple dataset for testing
    return DatasetDict({
        'train': Dataset.from_dict({
            'question': ['What is your name?'],
            'context': ['My name is Test.'],
            'answers': [{'answer_start': [11], 'text': ['Test']}],
            'id': [0],
        }),
        'validation': Dataset.from_dict({
            'question': ['What is the color of the sky?'],
            'context': ['The sky is blue.'],
            'answers': [{'answer_start': [12], 'text': ['blue']}],
            'id': [1],
        }),
    })


def test_data_preprocessor_init(mock_data_args, mock_tokenizer, mock_datasets):
    # DataPreProcessor 객체가 제대로 초기화되는지 검증합니다.
    with patch('datasets.load_from_disk', return_value=mock_datasets):
        data_processor = DataPreProcessor(mock_data_args, mock_tokenizer)
        assert data_processor.data_args == mock_data_args
        assert data_processor.tokenizer == mock_tokenizer
        assert data_processor.datasets == mock_datasets


def test_load_data_train(mock_data_args, mock_tokenizer, mock_datasets):
    # load_data 메소드가 'train' 타입으로 호출될 때의 결과를 검증합니다.
    with patch('datasets.load_from_disk', return_value=mock_datasets):
        data_processor = DataPreProcessor(mock_data_args, mock_tokenizer)
        processed_data = data_processor.load_data('train')

        assert 'input_ids' in processed_data
        assert processed_data['input_ids'][0] == [101, 102]
        assert len(processed_data['start_positions']) == 1
        assert processed_data['start_positions'][0] == 0


def test_load_data_validation(mock_data_args, mock_tokenizer, mock_datasets):
    # load_data 메소드가 'validation' 타입으로 호출될 때의 결과를 검증합니다.
    with patch('datasets.load_from_disk', return_value=mock_datasets):
        data_processor = DataPreProcessor(mock_data_args, mock_tokenizer)
        processed_data = data_processor.load_data('validation')

        assert 'input_ids' in processed_data
        assert processed_data['input_ids'][0] == [101, 102]
        assert 'example_id' in processed_data
        assert processed_data['example_id'][0] == 1


def test_load_data_invalid_type(mock_data_args, mock_tokenizer):
    # 잘못된 타입을 사용할 경우 예외가 발생하는지 검증합니다.
    with patch('datasets.load_from_disk', return_value=None):
        data_processor = DataPreProcessor(mock_data_args, mock_tokenizer)
        with pytest.raises(ValueError):
            data_processor.load_data('invalid_type')


def test_handle_features_train(mock_data_args, mock_tokenizer, mock_datasets):
    # handle_features 메소드가 'train' 타입으로 호출될 때의 결과를 검증합니다.
    with patch('datasets.load_from_disk', return_value=mock_datasets):
        data_processor = DataPreProcessor(mock_data_args, mock_tokenizer)
        processed_features = data_processor.handle_features(mock_datasets, 'train')

        assert 'input_ids' in processed_features
        assert len(processed_features['input_ids']) == 1


def test_handle_features_validation(mock_data_args, mock_tokenizer, mock_datasets):
    # handle_features 메소드가 'validation' 타입으로 호출될 때의 결과를 검증합니다.
    with patch('datasets.load_from_disk', return_value=mock_datasets):
        data_processor = DataPreProcessor(mock_data_args, mock_tokenizer)
        processed_features = data_processor.handle_features(mock_datasets, 'validation')

        assert 'input_ids' in processed_features
        assert len(processed_features['input_ids']) == 1


def test_data_post_processor(mock_data_args, mock_tokenizer):
    # DataPostProcessor 클래스의 handle_features 메소드를 검증합니다.
    data_post_processor = DataPostProcessor(mock_data_args, mock_tokenizer)
    processed_data = data_post_processor.handle_features({}, 'train')

    assert processed_data == {'data': 'post-processed'}
