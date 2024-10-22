from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from transformers import AutoTokenizer

from src import DataHandler
from src import DataProcessor
from src import DataTrainingArguments


@pytest.fixture
def data_handler():
    data_args = DataTrainingArguments(dataset_name='data/train_dataset')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    preprocessor = MagicMock(spec=DataProcessor)
    postprocessor = MagicMock(spec=DataProcessor)

    handler = DataHandler(data_args, tokenizer, postprocessor, preprocessor)

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
    preprocessor.process.return_value = 'processed_data'
    processed = handler.process_data('pre', None, 'train')
    assert processed == 'processed_data'
    preprocessor.process.assert_called_once()


def test_process_func(data_handler):
    handler, preprocessor, _ = data_handler
    preprocessor.process = MagicMock(return_value='processed_func')
    result = handler.process_func('pre')
    assert result() == 'processed_func'
