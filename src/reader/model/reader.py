from __future__ import annotations

from datasets import Dataset
from evaluate import load
from transformers import EvalPrediction
from transformers import TrainingArguments

from src.reader.data_controller.data_handler import DataHandler
from src.reader.data_controller.data_processor import DataPostProcessor
from src.reader.data_controller.data_processor import DataPreProcessor
from src.reader.model.huggingface_manager import HuggingFaceLoadManager
from src.reader.model.result_saver import ResultSaver
from src.reader.model.trainer_manager import TrainerManager
from src.utils.argument_validator import validate_flags
from src.utils.arguments import DataTrainingArguments
from src.utils.arguments import ModelArguments
from src.utils.log.logger import setup_logger


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
        self.data_args = data_args

        self.training_args = training_args

        '''성능향상 시도
        self.training_args.fp16 = True
        self.training_args.torch_compile = True'''

        self.datasets = datasets
        self.result_saver = ResultSaver(training_args, self.logger)
        self.metric = load(model_args.metric)

        self.data_handler = DataHandler(
            data_args=data_args, train_args=training_args,
            tokenizer=self.model_manager.get_tokenizer(),
            datasets=datasets,
            preprocessor=DataPreProcessor,                          # type: ignore[arg-type]
            postprocessor=DataPostProcessor,                        # type: ignore[arg-type]

        )

    def run(self):
        """Reader 실행 함수."""
        training_args = self.training_args
        data_handler = self.data_handler

        # 플래그 검증 추가
        validate_flags(training_args.do_train, training_args.do_eval, training_args.do_predict)

        # 데이터 전처리
        train_dataset = data_handler.load_data('train') if training_args.do_train else None
        eval_dataset = data_handler.load_data('validation') if training_args.do_eval else None
        test_dataset = data_handler.load_data('validation') if training_args.do_predict else None
        # LOOK do_eval이랑 do_predict 동시에 안될거 같다...?

        # TrainerManager 생성 및 실행
        trainer_manager = TrainerManager(
            model=self.model_manager.get_model(),
            tokenizer=self.model_manager.get_tokenizer(),
            training_args=training_args,
            data_args=self.data_args,
            compute_metrics=self.compute_metrics,
        )
        trainer = trainer_manager.create_trainer(
            train_dataset=train_dataset, eval_dataset=eval_dataset, eval_example=data_handler.plain_data(
                'validation',
            ) if training_args.do_eval or training_args.do_predict else None, post_processing_function=self.data_handler.process_func('pos'),
        )

        if training_args.do_train:
            train_result = trainer_manager.run_training(trainer, train_dataset)
            self.result_saver.save_results(trainer, train_result, train_dataset, 'train')

        if training_args.do_eval:
            eval_metrics = trainer_manager.run_evaluation(trainer, eval_dataset)
            self.logger.info(f'Evaluation results: {eval_metrics}')

        if training_args.do_predict:
            predictions = trainer_manager.run_prediction(trainer, test_dataset)
            self.result_saver.save_predictions(predictions)

    def compute_metrics(self, p: EvalPrediction):
        return self.metric.compute(predictions=p.predictions, references=p.label_ids)
