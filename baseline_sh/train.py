import logging
import os
import sys
import random
import numpy as np
import torch
import collections
import json
from typing import Tuple, Optional
from tqdm import tqdm
from arguments import DataTrainingArguments, ModelArguments
from datasets import load_from_disk, load_metric
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
        self.config = AutoConfig.from_pretrained(
            model_args.config_name or model_args.model_name_or_path
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name or model_args.model_name_or_path,
            use_fast=True,
        )
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=self.config,
        )
        self.data_collator = DataCollatorWithPadding(
            self.tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
        )
        self.preprocessor = DataPreprocessor(self.tokenizer, self.data_args, self.training_args)
        self.post_processor = PostProcessor()  # PostProcessor 인스턴스 생성


    def set_random_seed(self):
        seed_setter = SeedSetter(seed=2024)  # SeedSetter 인스턴스 생성
        seed_setter.set_seed()  # 시드 설정 메서드 호출

    def prepare_datasets(self):
        return self.preprocessor.prepare_datasets(self.datasets)


    def train(self):
        train_dataset, eval_dataset, last_checkpoint = self.prepare_datasets()

        # Trainer 초기화
        trainer = QuestionAnsweringTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset if self.training_args.do_train else None,
            eval_dataset=eval_dataset if self.training_args.do_eval else None,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            post_process_function=self.post_processing_function,
            compute_metrics=self.compute_metrics,
        )

        if self.training_args.do_train:
            checkpoint = last_checkpoint or self.model_args.model_name_or_path
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()
            self.log_results(train_result, "train")

        if self.training_args.do_eval:
            metrics = trainer.evaluate()
            self.log_results(metrics, "eval")

    def post_processing_function(self, examples, features, predictions: Tuple[np.ndarray, np.ndarray], version_2_with_negative: bool = False, n_best_size: int = 20, max_answer_length: int = 30, null_score_diff_threshold: float = 0.0, output_dir: Optional[str] = None, prefix: Optional[str] = None, is_world_process_zero: bool = True):
        """
        Post-processes predictions using PostProcessor.
        """
        return self.post_processor.process_predictions(
            examples, features, predictions,
            version_2_with_negative, n_best_size,
            max_answer_length, null_score_diff_threshold,
            output_dir, prefix, is_world_process_zero
        )

    def compute_metrics(self, p: EvalPrediction):
        metric = load_metric("squad")
        metrics = metric.compute(predictions=p.predictions, references=p.label_ids)
        return metrics
    
    def log_results(self, metrics, mode: str):
        output_file = os.path.join(self.training_args.output_dir, f"{mode}_results.txt")
        with open(output_file, "w") as writer:
            self.logger.info(f"***** {mode.capitalize()} results *****")
            for key, value in sorted(metrics.items()):
                self.logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")


def main():
    # Arguments 파싱
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    mrc_trainer = MRCTrainer(model_args, data_args, training_args)
    mrc_trainer.train()


if __name__ == "__main__":
    main()
