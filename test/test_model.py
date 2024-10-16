from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from datasets import DatasetDict
from transformers import PreTrainedTokenizerFast
from transformers import TrainingArguments

from src.reader.model import Reader
from src.reader.utils.arguments import DataTrainingArguments
from src.reader.utils.arguments import ModelArguments


@pytest.fixture
def mock_datasets():
    mock_train = MagicMock()
    mock_train.num_rows = 10
    mock_train.column_names = ['input_ids', 'attention_mask', 'labels']
    mock_train.map.return_value = mock_train
    mock_train.__getitem__.side_effect = lambda idx: {
        'input_ids': [1, 2, 3, 4],
        'attention_mask': [1, 1, 1, 1],
        'labels': 0,
    } if idx < mock_train.num_rows else KeyError(idx)
    mock_train.__iter__.return_value = iter([{
        'input_ids': [1, 2, 3, 4],
        'attention_mask': [1, 1, 1, 1],
        'labels': 0,
    } for _ in range(mock_train.num_rows)])

    mock_validation = MagicMock()
    mock_validation.num_rows = 5
    mock_validation.column_names = ['input_ids', 'attention_mask', 'labels']
    mock_validation.map.return_value = mock_validation
    mock_validation.__getitem__.side_effect = lambda idx: {
        'input_ids': [1, 2, 3, 4],
        'attention_mask': [1, 1, 1, 1],
        'labels': 0,
    } if idx < mock_validation.num_rows else KeyError(idx)
    mock_validation.__iter__.return_value = iter([{
        'input_ids': [1, 2, 3, 4],
        'attention_mask': [1, 1, 1, 1],
        'labels': 0,
    } for _ in range(mock_validation.num_rows)])

    return DatasetDict({
        'train': mock_train,
        'validation': mock_validation,
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


def test_load(mock_model_args, mock_data_args, mock_training_args, mock_datasets):
    reader = Reader(
        model_args=mock_model_args,
        data_args=mock_data_args,
        training_args=mock_training_args,
        datasets=mock_datasets,
    )

    with patch('transformers.AutoConfig.from_pretrained') as mock_config, \
        patch(
            'transformers.AutoTokenizer.from_pretrained',
            return_value=MagicMock(
                spec=PreTrainedTokenizerFast,
                model_max_length=512,
            ),
    ) as mock_tokenizer, \
            patch('transformers.AutoModelForQuestionAnswering.from_pretrained') as mock_model, \
            patch('src.reader.utils.tokenizer_checker.check_no_error', return_value=(None, 384)):

        reader.load()
        mock_config.assert_called_once_with('bert-base-uncased')
        mock_tokenizer.assert_called_once_with('bert-base-uncased', use_fast=True)
        mock_model.assert_called_once_with('bert-base-uncased', from_tf=False, config=mock_config())


def test_run_train(mock_model_args, mock_data_args, mock_training_args, mock_datasets):
    reader = Reader(
        model_args=mock_model_args,
        data_args=mock_data_args,
        training_args=mock_training_args,
        datasets=mock_datasets,
    )

    reader.load()

    with patch(
        'src.reader.trainer_qa.QuestionAnsweringTrainer.train',
        return_value=MagicMock(metrics={'accuracy': 0.9}),
    ) as mock_train, \
            patch('src.reader.trainer_qa.QuestionAnsweringTrainer.save_model') as mock_save_model, \
            patch('src.reader.data_processor.DataPreProcessor.process', return_value=mock_datasets['train']):

        reader._run_training = MagicMock()  # 내부 메서드 호출 모킹
        reader.run()
        reader._run_training.assert_called_once()

        mock_train.assert_called_once()
        mock_save_model.assert_called_once()


def test_run_predict(mock_model_args, mock_data_args, mock_training_args, mock_datasets):
    mock_training_args.do_predict = True  # Predict 모드를 활성화
    mock_datasets['train'].num_rows = 10
    mock_datasets['train'].map.return_value = mock_datasets['train']

    reader = Reader(
        model_args=mock_model_args,
        data_args=mock_data_args,
        training_args=mock_training_args,
        datasets=mock_datasets,
    )

    reader.load()

    with patch(
        'src.reader.trainer_qa.QuestionAnsweringTrainer.predict',
        return_value={'predictions': 'test_predictions'},
    ) as mock_predict, \
            patch('src.reader.data_processor.DataPreProcessor.process', return_value=mock_datasets['validation']):

        reader._run_prediction = MagicMock()  # 내부 메서드 호출 모킹
        reader.run()

        reader._run_prediction.assert_called_once()
        mock_predict.assert_called_once()


def test_save_results(mock_model_args, mock_data_args, mock_training_args, mock_datasets):
    reader = Reader(
        model_args=mock_model_args,
        data_args=mock_data_args,
        training_args=mock_training_args,
        datasets=mock_datasets,
    )

    with patch('builtins.open', new_callable=MagicMock) as mock_open, \
            patch('src.reader.trainer_qa.QuestionAnsweringTrainer.save_model'), \
            patch('src.reader.trainer_qa.QuestionAnsweringTrainer.log_metrics'), \
            patch('src.reader.trainer_qa.QuestionAnsweringTrainer.save_metrics'), \
            patch('src.reader.trainer_qa.QuestionAnsweringTrainer.save_state'):

        result = MagicMock(metrics={'accuracy': 0.9})
        reader._save_results(MagicMock(), result, mock_datasets['train'], 'train')

        mock_open.assert_called_once_with('./results/train_results.txt', 'w')
