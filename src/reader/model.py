import os
from typing import Any
from typing import Optional

from data_processor import DataPostProcessor
from data_processor import DataPreProcessor
from datasets import Dataset
from datasets import DatasetDict
from evaluate import load
from log.logger import setup_logger
from reader.utils.arguments import DataTrainingArguments
from reader.utils.arguments import ModelArguments
from trainer_qa import QuestionAnsweringTrainer
from transformers import AutoConfig
from transformers import AutoModelForQuestionAnswering
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from utils.seed import set_seed
from utils.tokenizer_checker import check_no_error


class Reader:
    # TODO: 리더 클래스 개발, DataHandler 변경 사항에 맞게 개발
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataTrainingArguments,
        training_args: TrainingArguments,
        datasets: DatasetDict,
    ) -> None:
        self.logger = setup_logger(__name__)
        set_seed()

        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.datasets = datasets

        if self.training_args.do_train:
            self.column_names = self.datasets['train'].column_names
        else:
            self.column_names = self.datasets['validation'].column_names

        self.output: dict = {}

    def load(self):
        self.config = AutoConfig.from_pretrained(
            self.model_args.config_name
            if self.model_args.config_name is not None
            else self.model_args.model_name_or_path,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name
            if self.model_args.tokenizer_name is not None
            else self.model_args.model_name_or_path,
            use_fast=True,
        )
        self.pad_on_right = self.tokenizer.padding_side == 'right'
        self.last_checkpoint, self.max_seq_length = check_no_error(
            data_args=self.data_args,
            training_args=self.training_args,
            datasets=self.datasets,
            tokenizer=self.tokenizer,
        )

        self.model = AutoModelForQuestionAnswering.from_pretrained(
            self.model_args.model_name_or_path,
            from_tf=bool('.ckpt' in self.model_args.model_name_or_path),
            config=self.config,
        )

    def preprocess_data(self, stage: str) -> Dataset:
        if stage == 'train':
            column_names = self.datasets['train'].column_names
            dataset = self.datasets['train']
        else:
            column_names = self.datasets['validation'].column_names
            dataset = self.datasets['validation']

        processed_dataset: Dataset = dataset.map(
            function=DataPreProcessor.process,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not self.data_args.overwrite_cache,
        )
        return processed_dataset

    def run(self) -> None:
        """훈련과 예측을 하나의 함수로 처리"""
        train_dataset: Optional[Dataset] = None
        eval_dataset: Optional[Dataset] = None

        if self.training_args.do_train:
            train_dataset = self.preprocess_data('train')
        if self.training_args.do_eval or self.training_args.do_predict:
            eval_dataset = self.preprocess_data('eval')

        self._run(train_dataset, eval_dataset)

    def _run(self, train_dataset: Optional[Dataset], eval_dataset: Optional[Dataset]) -> None:
        """훈련과 평가, 예측을 모두 포함한 함수"""
        data_collator: DataCollatorWithPadding = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8 if self.training_args.fp16 else None,
        )
        metric = load('squad')

        def compute_metrics(p: Any) -> dict[str, Any]:
            return metric.compute(predictions=p.predictions, references=p.label_ids)

        trainer: QuestionAnsweringTrainer = QuestionAnsweringTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset if self.training_args.do_train else None,
            eval_dataset=eval_dataset if self.training_args.do_eval else None,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            post_process_function=DataPostProcessor.process,
            compute_metrics=compute_metrics,
        )

        if self.training_args.do_train:
            self._run_training(trainer, train_dataset)
        if self.training_args.do_eval:
            self._run_evaluation(trainer, eval_dataset)
        if self.training_args.do_predict:
            self._run_prediction(trainer, eval_dataset)

    def _run_training(self, trainer: QuestionAnsweringTrainer, train_dataset: Dataset) -> None:
        """훈련 수행"""
        checkpoint: Optional[str]
        if self.last_checkpoint:
            checkpoint = self.last_checkpoint
        else:
            checkpoint = None

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        self._save_results(trainer, train_result, train_dataset, 'train')

    def _run_evaluation(self, trainer: QuestionAnsweringTrainer, eval_dataset: Dataset) -> None:
        """평가 수행"""
        self.logger.info('*** Evaluate ***')
        metrics = trainer.evaluate()
        metrics['eval_samples'] = len(eval_dataset)
        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)

    def _run_prediction(self, trainer: QuestionAnsweringTrainer, eval_dataset: Dataset) -> None:
        """예측 수행"""
        predictions = trainer.predict(eval_dataset)
        self._save_predictions(predictions)

    def _save_results(self, trainer: QuestionAnsweringTrainer, result: Any, dataset: Dataset, stage: str) -> None:
        """결과 저장"""
        trainer.save_model()
        metrics: dict[str, Any] = result.metrics
        metrics[f'{stage}_samples'] = len(dataset)

        trainer.log_metrics(stage, metrics)
        trainer.save_metrics(stage, metrics)
        trainer.save_state()

        output_file: str = os.path.join(self.training_args.output_dir, f'{stage}_results.txt')
        with open(output_file, 'w') as writer:
            self.logger.info(f'***** {stage.capitalize()} results *****')
            for key, value in sorted(metrics.items()):
                self.logger.info(f'{key} = {value}')
                writer.write(f'{key} = {value}\n')

    def _save_predictions(self, predictions: Any) -> None:
        """예측 결과 저장"""
        output_file = os.path.join(self.training_args.output_dir, 'predictions.json')
        with open(output_file, 'w') as writer:
            writer.write(predictions)
        self.logger.info(f'Predictions saved to {output_file}')
