from __future__ import annotations

from typing import Callable

from datasets import Dataset
from datasets import DatasetDict
from datasets import load_from_disk
from transformers import AutoTokenizer
from transformers import TrainingArguments

from src.reader.data_controller.data_processor import DataProcessor
from src.utils.arguments import DataTrainingArguments

class DataHandler():
    def __init__(self, data_args: DataTrainingArguments, train_args: TrainingArguments, tokenizer: AutoTokenizer, datasets: Dataset, postprocessor: DataProcessor, preprocessor: DataProcessor) -> None:
        """DataHandler 초기화 설정.
        Args:
            data_args (DataTrainingArguments): DataTrainingArguments 형식
            tokenizer (AutoTokenizer): 토크나이저 설정
        """
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.max_seq_length = min(
            data_args.max_seq_length, tokenizer.model_max_length,
        )

        self.data_args.output_dir = train_args.output_dir           # type: ignore[attr-defined]
        self.data_args.do_predict = train_args.do_predict           # type: ignore[attr-defined]
        self.data_args.do_eval = train_args.do_eval                 # type: ignore[attr-defined]
        self.data_args.do_train = train_args.do_train               # type: ignore[attr-defined]

        if datasets:
            self.datasets = datasets
        else:
            self.datasets = load_from_disk(data_args.dataset_name)

        self.processors = {
            preprocessor.name: preprocessor,
            postprocessor.name: postprocessor,
        }

    def process_func(self, type: str) -> Callable:
        """데이터를 처리하는 함수를 반환

        Args:
            type (str): 처리할 dataprocessor 정보

        Returns:
            processed
        """
        processor_func = self.processors[type].process
        return processor_func

    def process_data(self, proc: str, data_type: str):
        """데이터를 처리함

        Args:
            type (str): _description_

        Returns:
            BatchEncoding: _description_
        """
        processed_data = self.processors[proc].process(self.data_args, self.datasets[data_type], tokenizer=self.tokenizer)
        return processed_data

    def load_data(self, type: str) -> DatasetDict:
        """데이터를 로드, 기본적으로 전처리 함

        Args:
            type (str): 로드할 데이터 종류를 지정, 'train'/'validation'

        Returns:
            datasets
        """
        if type not in ['train', 'validation']:
            raise ValueError("Invalid type: must be 'train' or 'validation'")

        if type not in self.datasets:
            raise ValueError('--do_'+type+' requires a '+type+' dataset')

        datasets = self.process_data('pre', type)

        return datasets

    def plain_data(self, type) -> DatasetDict:
        """전처리 거치지 않은 데이터셋을 반환

        Args:
            type (str): data type
        """
        return self.datasets[type]
