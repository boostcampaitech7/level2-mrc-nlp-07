from __future__ import annotations

from datasets import Dataset
from evaluate import load
from transformers import TrainingArguments

from reader.data_controller.data_handler import DataHandler
from reader.data_controller.data_processor import DataPostProcessor
from reader.data_controller.data_processor import DataPreProcessor
from reader.model.huggingface_manager import HuggingFaceLoadManager
from reader.model.result_saver import ResultSaver
from reader.model.trainer_manager import TrainerManager
from utils.argument_validator import validate_flags
from utils.arguments import DataTrainingArguments, ModelArguments
from utils.log.logger import setup_logger


class Reader:
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataTrainingArguments,
        training_args: TrainingArguments,
        datasets: Dataset = None,
    ):
        self.logger = setup_logger(model_args.model_name_or_path)
        self.model_manager = HuggingFaceLoadManager(model_args)

        self.data_handler = DataHandler(
            data_args=data_args, train_args=training_args,
            tokenizer=self.model_manager.get_tokenizer(),
            preprocessor=DataPreProcessor,      # type: ignore
            postprocessor=DataPostProcessor,    # type: ignore
        )
        self.training_args = training_args
        self.datasets = datasets
        self.result_saver = ResultSaver(training_args, self.logger)

    def run(self):
        """Reader 실행 함수."""
        # 플래그 검증 추가
        validate_flags(self.training_args.do_train, self.training_args.do_eval, self.training_args.do_predict)

        # 데이터 전처리
        train_dataset = self.data_handler.load_data('train') if self.training_args.do_train else None
        eval_dataset = self.data_handler.load_data('validation') if self.training_args.do_eval else None
        test_dataset = self.data_handler.load_data('validation') if self.training_args.do_predict else None

        # TrainerManager 생성 및 실행
        trainer_manager = TrainerManager(
            model=self.model_manager.get_model(),
            tokenizer=self.model_manager.get_tokenizer(),
            training_args=self.training_args,
            compute_metrics=lambda p: load('squad').compute(predictions=p.predictions, references=p.label_ids),
        )

        trainer = trainer_manager.create_trainer(train_dataset, eval_dataset)

        if self.training_args.do_train:
            train_result = trainer_manager.run_training(trainer, train_dataset)
            self.result_saver.save_results(trainer, train_result, train_dataset, 'train')

        if self.training_args.do_eval:
            eval_metrics = trainer_manager.run_evaluation(trainer, eval_dataset)
            self.logger.info(f'Evaluation results: {eval_metrics}')

        if self.training_args.do_predict:
            predictions = trainer_manager.run_prediction(trainer, test_dataset, eval_dataset)
            self.result_saver.save_predictions(predictions)
