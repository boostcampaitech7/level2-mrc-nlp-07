from reader.model import Reader
from reader.monitoring_tool import MonitoringTool
from reader.utils.arguments import DataTrainingArguments
from reader.utils.arguments import ModelArguments
from transformers import HfArgumentParser
from transformers import TrainingArguments


def main():
    # TODO: 모델 학습 과정 추가 생성
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments),
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    reader_model = Reader(model_args, data_args, training_args)
    reader_model.load()

    # evaluation = Evaluation()
    monitor_tool = MonitoringTool()

    monitor_tool.start_monitoring()

    reader_model.train()

    monitor_tool.stop_monitoring()


if __name__ == '__main__':
    main()
