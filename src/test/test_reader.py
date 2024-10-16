from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from datasets import DatasetDict
from transformers import TrainingArguments

from ..reader.model import Reader
from ..reader.utils.arguments import DataTrainingArguments
from ..reader.utils.arguments import ModelArguments
# Mock된 DataTrainingArguments와 ModelArguments


@pytest.fixture
def mock_datasets():
    return DatasetDict({
        'train': MagicMock(),
        'validation': MagicMock(),
    })


@pytest.fixture
def mock_model_args():
    return ModelArguments(
        model_name_or_path='bert-base-uncased',
        config_name=None,
        tokenizer_name=None,
    )


@pytest.fixture
def mock_data_args():
    return DataTrainingArguments(
        max_seq_length=384,
        overwrite_cache=False,
        preprocessing_num_workers=4,
        pad_to_max_length=True,
    )


@pytest.fixture
def mock_training_args():
    return TrainingArguments(
        output_dir='./results',
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        logging_dir='./logs',
    )

# load() 메소드 테스트


def test_load(mock_model_args, mock_data_args, mock_training_args, mock_datasets):
    reader = Reader(
        model_args=mock_model_args,
        data_args=mock_data_args,
        training_args=mock_training_args,
        datasets=mock_datasets,
    )

    with patch('transformers.AutoConfig.from_pretrained') as mock_config, \
            patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
            patch('transformers.AutoModelForQuestionAnswering.from_pretrained') as mock_model, \
            patch('utils.tokenizer_checker.check_no_error', return_value=(None, 384)):

        reader.load()
        mock_config.assert_called_once_with('bert-base-uncased')
        mock_tokenizer.assert_called_once_with('bert-base-uncased', use_fast=True)
        mock_model.assert_called_once_with('bert-base-uncased', from_tf=False, config=mock_config())

# train() 메소드 테스트


def test_train(mock_model_args, mock_data_args, mock_training_args, mock_datasets):
    reader = Reader(
        model_args=mock_model_args,
        data_args=mock_data_args,
        training_args=mock_training_args,
        datasets=mock_datasets,
    )

    reader.load()

    with patch(
        'trainer_qa.QuestionAnsweringTrainer.train',
        return_value=MagicMock(metrics={'accuracy': 0.9}),
    ) as mock_train, \
            patch('trainer_qa.QuestionAnsweringTrainer.save_model') as mock_save_model:

        reader.train()
        mock_train.assert_called_once()
        mock_save_model.assert_called_once()

# predict() 메소드 테스트


def test_predict(mock_model_args, mock_data_args, mock_training_args, mock_datasets):
    reader = Reader(
        model_args=mock_model_args,
        data_args=mock_data_args,
        training_args=mock_training_args,
        datasets=mock_datasets,
    )

    output = {'answer': 'test_answer'}
    predictions = reader.predict(output)
    assert predictions['predictions'] == output
