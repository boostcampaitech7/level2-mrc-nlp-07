from __future__ import annotations

from transformers import HfArgumentParser
from transformers import TrainingArguments

from src import Reader
from src import DataTrainingArguments, ModelArguments


def main():
    # TODO: 모델 학습 과정 추가 생성
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments),
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    reader_model = Reader(model_args, data_args, training_args)
    reader_model.run()


if __name__ == '__main__':
    main()
