from __future__ import annotations

import os
from typing import Any

from datasets import DatasetDict
from transformers import PreTrainedTokenizerFast
from transformers import TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from .log.logger import setup_logger
from src.utils.arguments import DataTrainingArguments


def check_no_error(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    datasets: DatasetDict,
    tokenizer: PreTrainedTokenizerFast,
) -> tuple[Any, int]:
    last_checkpoint = _find_last_checkpoint(training_args)
    _validate_tokenizer(tokenizer, data_args.max_seq_length)
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if 'validation' not in datasets:
        raise ValueError('--do_eval requires a validation dataset')

    return last_checkpoint, max_seq_length


def _find_last_checkpoint(training_args: TrainingArguments) -> str | None:
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and os.listdir(training_args.output_dir):
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to train from scratch.',
            )
        return last_checkpoint
    return None


def _validate_tokenizer(tokenizer: PreTrainedTokenizerFast, max_seq_length: int):
    logger = setup_logger(__name__)

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError('Fast tokenizer required for this script.')
    if max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"max_seq_length ({max_seq_length}) exceeds model's max length ({tokenizer.model_max_length}). "
            f'Using max_seq_length={tokenizer.model_max_length}.',
        )
