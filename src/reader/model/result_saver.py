from __future__ import annotations

import os
from logging import Logger
from typing import Any

from datasets import Dataset
from transformers import TrainingArguments

from reader.model.trainer_qa import QuestionAnsweringTrainer


class ResultSaver:
    def __init__(self, training_args: TrainingArguments, logger: Logger):
        self.training_args = training_args
        self.logger = logger

    def save_results(
        self,
        trainer: QuestionAnsweringTrainer,
        result: Any,
        dataset: Dataset,
        stage: str,
    ):
        trainer.save_model()
        metrics = result.metrics
        metrics[f'{stage}_samples'] = len(dataset)
        trainer.log_metrics(stage, metrics)
        trainer.save_metrics(stage, metrics)
        trainer.save_state()

        output_file = os.path.join(self.training_args.output_dir, f'{stage}_results.txt')
        with open(output_file, 'w') as writer:
            self.logger.info(f'***** {stage.capitalize()} results *****')
            for key, value in sorted(metrics.items()):
                self.logger.info(f'{key} = {value}')
                writer.write(f'{key} = {value}\n')

    def save_predictions(self, predictions):
        output_file = os.path.join(self.training_args.output_dir, 'predictions.json')
        with open(output_file, 'w') as writer:
            writer.write(predictions)
        self.logger.info(f'Predictions saved to {output_file}')
