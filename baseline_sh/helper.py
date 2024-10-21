import logging
import os
import random
from typing import Any, Tuple, Optional

import numpy as np
import torch
from arguments import DataTrainingArguments
from datasets import DatasetDict
from transformers import PreTrainedTokenizerFast, TrainingArguments, is_torch_available
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)


class SeedSetter:
    def __init__(self, seed: int = 42):
        """
        SeedSetter 초기화.

        Args:
            seed (int): 설정할 시드 값.
        """
        self.seed = seed

    def set_seed(self) -> None:
        """
        seed 고정하는 메서드 (random, numpy, torch).
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        if is_torch_available():
            self._set_torch_seed()

    def _set_torch_seed(self) -> None:
        """
        Torch에 대한 시드를 설정하는 헬퍼 메서드.
        """
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)  # multi-GPU 사용할 경우
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TrainingValidator:
    def __init__(self, data_args: DataTrainingArguments, training_args: TrainingArguments, datasets: DatasetDict, tokenizer: PreTrainedTokenizerFast):
        self.data_args = data_args
        self.training_args = training_args
        self.datasets = datasets
        self.tokenizer = tokenizer

    def check_no_error(self) -> Tuple[Optional[str], int]:
        """
        오류가 없는지 체크하고 마지막 체크포인트와 최대 시퀀스 길이를 반환합니다.

        Returns:
            Tuple[Optional[str], int]: 마지막 체크포인트와 최대 시퀀스 길이.

        Raises:
            ValueError: 유효하지 않은 설정에 대한 오류.
        """
        last_checkpoint = self.find_last_checkpoint()
        logger.info("Validating tokenizer...")
        self.validate_tokenizer()
        logger.info("Validating max sequence length...")
        max_seq_length = self.validate_max_seq_length()
        logger.info("Validating validation dataset...")
        self.validate_validation_dataset()
        
        logger.info("All validations passed successfully.")
        return last_checkpoint, max_seq_length

    def find_last_checkpoint(self) -> Optional[str]:
        """
        마지막 체크포인트를 찾는 헬퍼 메서드.

        Returns:
            Optional[str]: 마지막 체크포인트 경로 또는 None.
        """
        last_checkpoint = None
        if (
            os.path.isdir(self.training_args.output_dir)
            and self.training_args.do_train
            and not self.training_args.overwrite_output_dir
        ):
            last_checkpoint = get_last_checkpoint(self.training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(self.training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({self.training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
        return last_checkpoint

    def validate_tokenizer(self) -> None:
        """
        토크나이저가 Fast tokenizer인지 검증하는 메서드.

        Raises:
            ValueError: Fast tokenizer가 아닐 경우.
        """
        if not isinstance(self.tokenizer, PreTrainedTokenizerFast):
            raise ValueError(
                "This example script only works for models that have a fast tokenizer. "
                "Checkout the big table of models at https://huggingface.co/transformers/index.html#bigtable "
                "to find the model types that meet this requirement."
            )

    def validate_max_seq_length(self) -> int:
        """
        max_seq_length가 토크나이저의 최대 길이를 초과하는지 검증하는 메서드.

        Returns:
            int: 최대 시퀀스 길이.
        """
        if self.data_args.max_seq_length > self.tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({self.data_args.max_seq_length}) is larger than the maximum length for the "
                f"model ({self.tokenizer.model_max_length}). Using max_seq_length={self.tokenizer.model_max_length}."
            )
        return min(self.data_args.max_seq_length, self.tokenizer.model_max_length)

    def validate_validation_dataset(self) -> None:
        """
        validation 데이터셋이 존재하는지 검증하는 메서드.

        Raises:
            ValueError: validation 데이터셋이 없을 경우.
        """
        if "validation" not in self.datasets:
            raise ValueError("--do_eval requires a validation dataset")