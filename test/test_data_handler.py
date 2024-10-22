from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from transformers import AutoTokenizer
from src.reader.data_controller.data_handler import DataHandler
from src.utils.arguments import DataTrainingArguments, ModelArguments
from src.reader.data_controller.data_processor import DataProcessor


@pytest.fixture
def data_handler():
    data_args = DataTrainingArguments(dataset_name="data/train_dataset")
    model_args = ModelArguments(output_dir="./models/train_dataset", do_predict=False, do_eval=False, do_train=True)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Mock 객체 생성
    preprocessor = MagicMock(spec=DataProcessor)
    postprocessor = MagicMock(spec=DataProcessor)

    # DataHandler 초기화 (model_args 추가)
    handler = DataHandler(data_args, model_args, tokenizer, postprocessor, preprocessor)

    return handler, preprocessor, postprocessor


def test_initialization(data_handler):
    handler, _, _ = data_handler
    assert handler.data_args is not None
    assert handler.tokenizer is not None
    assert 'pre' in handler.processors
    assert 'pos' in handler.processors


def test_load_data_invalid_type(data_handler):
    handler, _, _ = data_handler
    with pytest.raises(ValueError, match="Invalid type: must be 'train' or 'validation'"):
        handler.load_data('invalid_type')


def test_load_data_missing_dataset(data_handler):
    handler, _, _ = data_handler
    handler.datasets = {}
    with pytest.raises(ValueError, match='--do_train requires a train dataset'):
        handler.load_data('train')


def test_process_data(data_handler):
    handler, preprocessor, _ = data_handler
    preprocessor.process.return_value = "processed_data"
    processed = handler.process_data('pre', None, 'train')
    assert processed == "processed_data"
    preprocessor.process.assert_called_once()


def test_process_func(data_handler):
    handler, preprocessor, _ = data_handler
    preprocessor.process = MagicMock(return_value="processed_func")
    result = handler.process_func('pre')
    assert result() == "processed_func"
