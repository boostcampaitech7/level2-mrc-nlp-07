import logging
import os
import sys
import numpy as np
import torch
from typing import Tuple, Optional
from datasets import load_from_disk, load_metric
from tqdm import tqdm
from arguments import DataTrainingArguments, ModelArguments
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    TrainingArguments,
    HfArgumentParser,
)
from helper import SeedSetter
from data_preprocessing import DataPreprocessor
from post_processing import PostProcessor

class MRCTrainer:
    def __init__(self, model_args: ModelArguments, data_args: DataTrainingArguments, training_args: TrainingArguments):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.logger = logging.getLogger(__name__)

        self.set_random_seed()
        self.datasets = load_from_disk(data_args.dataset_name)

        self.model, self.tokenizer, self.data_collator = self.initialize_model_and_tokenizer()
        self.preprocessor = DataPreprocessor(self.tokenizer, self.data_args, self.training_args)

    def set_random_seed(self) -> None:
        seed_setter = SeedSetter(seed=2024)
        seed_setter.set_seed()

    def initialize_model_and_tokenizer(self) -> Tuple[torch.nn.Module, AutoTokenizer, DataCollatorWithPadding]:
        """모델과 토크나이저를 초기화하는 메소드."""
        config = AutoConfig.from_pretrained(
            self.model_args.config_name or self.model_args.model_name_or_path
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name or self.model_args.model_name_or_path,
            use_fast=True,
        )
        model = AutoModelForQuestionAnswering.from_pretrained(
            self.model_args.model_name_or_path,
            from_tf=".ckpt" in self.model_args.model_name_or_path,
            config=config,
        )
        data_collator = DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8 if self.training_args.fp16 else None
        )
        return model, tokenizer, data_collator

    def prepare_datasets(self) -> Tuple[Optional[torch.utils.data.Dataset], Optional[torch.utils.data.Dataset], Optional[str]]:
        """데이터셋을 준비하는 메소드."""
        return self.preprocessor.prepare_datasets(self.datasets)

    def train(self) -> None:
        """훈련 프로세스를 실행하는 메소드."""
        train_dataset, eval_dataset, last_checkpoint = self.prepare_datasets()
        trainer = self.initialize_trainer(train_dataset, eval_dataset)

        if self.training_args.do_train:
            self.start_training(trainer, last_checkpoint, train_dataset)

        if self.training_args.do_eval:
            self.start_evaluation(trainer)

    def initialize_trainer(self, train_dataset, eval_dataset) -> QuestionAnsweringTrainer:
        """훈련기를 초기화하는 메소드."""
        return QuestionAnsweringTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset if self.training_args.do_train else None,
            eval_dataset=eval_dataset if self.training_args.do_eval else None,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            post_process_function=self.post_processing_function,
            compute_metrics=self.compute_metrics,
        )

    def start_training(self, trainer: QuestionAnsweringTrainer, last_checkpoint: Optional[str], train_dataset: Optional[torch.utils.data.Dataset]) -> None:
        """훈련 프로세스를 시작하는 메소드."""
        checkpoint = last_checkpoint if last_checkpoint is not None else (self.model_args.model_name_or_path if os.path.isdir(self.model_args.model_name_or_path) else None)

        self.logger.info("Starting training...")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        self.logger.info("Training finished.")

        self.log_training_metrics(train_result, train_dataset, trainer)  # trainer 추가

        trainer.save_model()

    def log_training_metrics(self, train_result, train_dataset: Optional[torch.utils.data.Dataset], trainer: QuestionAnsweringTrainer) -> None:
        """훈련 메트릭을 로깅하는 메소드."""
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        self.log_results(metrics, "train")

    def start_evaluation(self, trainer: QuestionAnsweringTrainer) -> None:
        """평가 프로세스를 시작하는 메소드."""
        self.logger.info("Starting evaluation...")
        metrics = trainer.evaluate()
        self.log_results(metrics, "eval")
        self.logger.info("Evaluation finished.")

    def post_processing_function(self, examples, features, predictions) -> dict:
        """후처리 함수를 정의하는 메소드."""
        postprocessor = PostProcessor(
            version_2_with_negative=False,
            n_best_size=20,
            max_answer_length=30,
            null_score_diff_threshold=0.0,
            output_dir=self.training_args.output_dir,
            prefix=None,
            is_world_process_zero=self.training_args.local_rank in [-1, 0]
        )

        predictions = postprocessor.post_process(examples, features, predictions)
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        if self.training_args.do_predict:
            return formatted_predictions

        elif self.training_args.do_eval:
            references = [{"id": ex["id"], "answers": ex[self.data_args.answer_column_name]} for ex in self.datasets["validation"]]
            return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    def compute_metrics(self, p: EvalPrediction) -> dict:
        """메트릭을 계산하는 메소드."""
        metric = load_metric("squad")
        metrics = metric.compute(predictions=p.predictions, references=p.label_ids)
        return metrics

    def log_results(self, metrics: dict, mode: str) -> None:
        """결과를 로깅하는 메소드."""
        output_file = os.path.join(self.training_args.output_dir, f"{mode}_results.txt")
        with open(output_file, "w") as writer:
            self.logger.info(f"***** {mode.capitalize()} results *****")
            for key, value in sorted(metrics.items()):
                self.logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

def main() -> None:
    """메인 함수."""
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    mrc_trainer = MRCTrainer(model_args, data_args, training_args)
    mrc_trainer.train()

if __name__ == "__main__":
    main()
