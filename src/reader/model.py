import logging
import os

from arguments import DataTrainingArguments
from arguments import ModelArguments
from data_processor import DataPostProcessor
from data_processor import DataPreProcessor
from datasets import DatasetDict
from evaluate import load
from trainer_qa import QuestionAnsweringTrainer
from transformers import AutoConfig
from transformers import AutoModelForQuestionAnswering
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments


logger = logging.getLogger(__name__)


class Reader:
    # TODO: 리더 클래스 개발, DataHandler 변경 사항에 맞게 개발
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataTrainingArguments,
        training_args: TrainingArguments,
        datasets: DatasetDict,
    ) -> None:
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
        # TODO: check_no_error 함수 문제 해결
        self.last_checkpoint, self.max_seq_length = None, None
        # check_no_error(
        #     data_args=self.data_args,
        #     training_args=self.training_args,
        #     datasets=self.datasets,
        #     tokenizer=self.tokenizer,
        # )

        self.model = AutoModelForQuestionAnswering.from_pretrained(
            self.model_args.model_name_or_path,
            from_tf=bool('.ckpt' in self.model_args.model_name_or_path),
            config=self.config,
        )

    def train(self) -> None:
        if self.training_args.do_train:
            if 'train' not in self.datasets:
                raise ValueError('--do_train requires a train dataset')
            train_dataset = self.datasets['train']
            train_dataset = train_dataset.map(
                function=DataPreProcessor.process,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=self.column_names,
                load_from_cache_file=not self.data_args.overwrite_cache,
            )

        if self.training_args.do_eval:
            eval_dataset = self.datasets['validation']
            eval_dataset = eval_dataset.map(
                function=DataPreProcessor.process,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=self.column_names,
                load_from_cache_file=not self.data_args.overwrite_cache,
            )

        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8 if self.training_args.fp16 else None,
        )

        metric = load('squad')

        def compute_metrics(p): return metric.compute(
            predictions=p.predictions,
            references=p.label_ids,
        )

        trainer = QuestionAnsweringTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset if self.training_args.do_train else None,
            eval_dataset=eval_dataset if self.training_args.do_eval else None,
            eval_examples=self.datasets['validation'] if self.training_args.do_eval else None,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            post_process_function=DataPostProcessor.process,
            compute_metrics=compute_metrics,
        )

        # Training
        if self.training_args.do_train:
            if self.last_checkpoint is not None:
                checkpoint = self.last_checkpoint
            elif os.path.isdir(self.model_args.model_name_or_path):
                checkpoint = self.model_args.model_name_or_path
            else:
                checkpoint = None
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # Saves the tokenizer too for easy upload

            metrics = train_result.metrics
            metrics['train_samples'] = len(train_dataset)

            trainer.log_metrics('train', metrics)
            trainer.save_metrics('train', metrics)
            trainer.save_state()

            output_train_file = os.path.join(self.training_args.output_dir, 'train_results.txt')

            with open(output_train_file, 'w') as writer:
                logger.info('***** Train results *****')
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f'  {key} = {value}')
                    writer.write(f'{key} = {value}\n')

            # State 저장
            trainer.state.save_to_json(
                os.path.join(self.training_args.output_dir, 'trainer_state.json'),
            )

        # Evaluation
        if self.training_args.do_eval:
            logger.info('*** Evaluate ***')
            metrics = trainer.evaluate()

            metrics['eval_samples'] = len(eval_dataset)

            trainer.log_metrics('eval', metrics)
            trainer.save_metrics('eval', metrics)

    def predict(self, output: dict) -> dict:
        return {'predictions': output}
