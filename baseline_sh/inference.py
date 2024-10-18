import logging
import sys
from typing import Callable, Dict, List, NoReturn, Tuple, Optional

import numpy as np
from arguments import DataTrainingArguments, ModelArguments
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_from_disk,
    load_metric,
)
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from utils_qa import postprocess_qa_predictions
from helper import TrainingValidator

logger = logging.getLogger(__name__)

class InferenceRunner:
    def __init__(self):
        self.model_args: ModelArguments
        self.data_args: DataTrainingArguments
        self.training_args: TrainingArguments
        self.tokenizer = None
        self.model = None
        self.datasets = None

    def parse_arguments(self) -> None:
        parser = HfArgumentParser(
            (ModelArguments, DataTrainingArguments, TrainingArguments)
        )
        self.model_args, self.data_args, self.training_args = parser.parse_args_into_dataclasses()
        self.training_args.do_train = True

    def setup_logging(self) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        logger.info("Training/evaluation parameters %s", self.training_args)

    def load_model_and_tokenizer(self) -> None:
        config = AutoConfig.from_pretrained(
            self.model_args.config_name if self.model_args.config_name else self.model_args.model_name_or_path,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name if self.model_args.tokenizer_name else self.model_args.model_name_or_path,
            use_fast=True,
        )
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            self.model_args.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
            config=config,
        )

    def load_datasets(self) -> None:
        self.datasets = load_from_disk(self.data_args.dataset_name)
        logger.info(f"Loaded datasets from {self.data_args.dataset_name}")

    def run_inference(self) -> NoReturn:
        if self.training_args.do_eval or self.training_args.do_predict:
            self.run_mrc()

    def run_mrc(self) -> NoReturn:
        column_names = self.datasets["validation"].column_names
        question_column_name = "question" if "question" in column_names else column_names[0]
        context_column_name = "context" if "context" in column_names else column_names[1]
        answer_column_name = "answers" if "answers" in column_names else column_names[2]

        pad_on_right = self.tokenizer.padding_side == "right"
        last_checkpoint, max_seq_length = TrainingValidator.check_no_error(self.data_args, self.training_args, self.datasets, self.tokenizer)

        eval_dataset = self.prepare_validation_features(column_names, question_column_name, context_column_name, pad_on_right, max_seq_length)

        data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8 if self.training_args.fp16 else None)

        post_processing_function = self.create_post_processing_function(eval_dataset, answer_column_name)

        metric = load_metric("squad")
        compute_metrics = lambda p: metric.compute(predictions=p.predictions, references=p.label_ids)

        trainer = QuestionAnsweringTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=None,
            eval_dataset=eval_dataset,
            eval_examples=self.datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            post_process_function=post_processing_function,
            compute_metrics=compute_metrics,
        )

        logger.info("*** Evaluate ***")
        self.evaluate_or_predict(trainer)

    def prepare_validation_features(self, column_names: List[str], question_column_name: str, context_column_name: str, pad_on_right: bool, max_seq_length: int) -> Dataset:
        def prepare_validation_features_inner(examples):
            tokenized_examples = self.tokenizer(
                examples[question_column_name if pad_on_right else context_column_name],
                examples[context_column_name if pad_on_right else question_column_name],
                truncation="only_second" if pad_on_right else "only_first",
                max_length=max_seq_length,
                stride=self.data_args.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length" if self.data_args.pad_to_max_length else False,
            )

            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
            tokenized_examples["example_id"] = []

            for i in range(len(tokenized_examples["input_ids"])):
                sequence_ids = tokenized_examples.sequence_ids(i)
                context_index = 1 if pad_on_right else 0
                sample_index = sample_mapping[i]
                tokenized_examples["example_id"].append(examples["id"][sample_index])
                tokenized_examples["offset_mapping"][i] = [
                    (o if sequence_ids[k] == context_index else None)
                    for k, o in enumerate(tokenized_examples["offset_mapping"][i])
                ]
            return tokenized_examples

        eval_dataset = self.datasets["validation"].map(
            prepare_validation_features_inner,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not self.data_args.overwrite_cache,
        )
        return eval_dataset

    def create_post_processing_function(self, eval_dataset: Dataset, answer_column_name: str) -> Callable:
        def post_processing_function(examples, features, predictions: Tuple[np.ndarray, np.ndarray], training_args: TrainingArguments) -> EvalPrediction:
            predictions = postprocess_qa_predictions(
                examples=examples,
                features=features,
                predictions=predictions,
                max_answer_length=self.data_args.max_answer_length,
                output_dir=training_args.output_dir,
            )
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

            if training_args.do_predict:
                return formatted_predictions
            elif training_args.do_eval:
                references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in eval_dataset]
                return EvalPrediction(predictions=formatted_predictions, label_ids=references)
        
        return post_processing_function

    def evaluate_or_predict(self, trainer: QuestionAnsweringTrainer) -> None:
        if self.training_args.do_predict:
            predictions = trainer.predict(test_dataset=self.datasets["validation"], test_examples=self.datasets["validation"])
            print("No metric can be presented because there is no correct answer given. Job done!")
        if self.training_args.do_eval:
            metrics = trainer.evaluate()
            metrics["eval_samples"] = len(self.datasets["validation"])
            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)


def main() -> None:
    runner = InferenceRunner()
    runner.parse_arguments()
    runner.setup_logging()
    runner.load_model_and_tokenizer()
    runner.load_datasets()
    runner.run_inference()


if __name__ == "__main__":
    main()
