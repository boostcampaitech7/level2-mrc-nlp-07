from __future__ import annotations

from arguments import DataTrainingArguments
from arguments import ModelArguments
from reader.data_handler import DataHandler
from reader.evaluation import Evaluation
from reader.model import Reader
from reader.monitoring_tool import MonitoringTool
from transformers import Trainer
from transformers import TrainingArguments


def main():
    # TODO: 모델 학습 과정 추가 생성
    model_args = ModelArguments()
    data_args = DataTrainingArguments()
    training_args = TrainingArguments(
        output_dir='./results',
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    reader_model = Reader(model_args)
    reader_model.load_model()
    reader_model.load_tokenizer()

    data_handler = DataHandler(data_args)
    evaluation = Evaluation()
    monitor_tool = MonitoringTool()

    train_dataset = data_handler.load_data()
    eval_dataset = data_handler.load_data()

    monitor_tool.start_monitoring()

    trainer = Trainer(
        model=reader_model.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=evaluation.compute_metrics,
    )

    trainer.train()
    trainer.evaluate()

    monitor_tool.stop_monitoring()


if __name__ == '__main__':
    main()
