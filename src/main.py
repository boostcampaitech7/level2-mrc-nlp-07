from __future__ import annotations

from reader.model.reader import Reader
from transformers import HfArgumentParser
from transformers import TrainingArguments
from utils.arguments import DataTrainingArguments
from utils.arguments import ModelArguments


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # output_dir은 training_args에 설정된 값을 사용
    print('Output directory:', training_args.output_dir)

    reader_model = Reader(model_args, data_args, training_args)
    reader_model.run()


if __name__ == '__main__':
    main()
