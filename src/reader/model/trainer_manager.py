from __future__ import annotations

from typing import Any
from typing import Callable

from datasets import Dataset
from transformers import AutoModelForQuestionAnswering
from transformers import AutoTokenizer
from transformers import BatchEncoding
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments

from src.reader.data_controller.data_processor import DataPostProcessor
from src.reader.model.trainer_qa import QuestionAnsweringTrainer
from src.utils.arguments import DataTrainingArguments
# TODO 종속성 문제 존재, 추후 수정 필요


class TrainerManager:
    def __init__(
        self,
        model: AutoModelForQuestionAnswering,
        tokenizer: AutoTokenizer,
        training_args: TrainingArguments,
        data_args: DataTrainingArguments,
        compute_metrics: Callable,
    ):
        """
        TrainerManager는 모델 학습, 평가, 예측을 관리하는 클래스입니다.

        Args:
            model (Any): 학습 및 평가에 사용할 모델.
            tokenizer (Any): 데이터를 토크나이즈하는 토크나이저.
            training_args (TrainingArguments): 모델 학습을 위한 트레이닝 인자.
            compute_metrics (Optional[callable]): 평가 시 사용되는 메트릭 계산 함수.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.compute_metrics = compute_metrics

        self.training_args.max_answer_length = data_args.max_answer_length
        # TODO trainig_args에 max_answer_length 동적 할당

    def create_trainer(
        self,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        eval_example: Dataset | None = None,
        post_processing_function: Callable | None = DataPostProcessor.process,
    ) -> QuestionAnsweringTrainer:
        """
        모델 학습 및 평가를 위한 Trainer를 생성합니다.

        Args:
            train_dataset (Optional[Dataset]): 학습에 사용할 데이터셋.
            eval_dataset (Optional[Dataset]): 평가에 사용할 데이터셋.

        Returns:
            QuestionAnsweringTrainer: 생성된 Trainer 객체.
        """
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8 if self.training_args.fp16 else None,
        )
        return QuestionAnsweringTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            eval_examples=eval_example,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            post_process_function=post_processing_function,
        )

    def run_training(
        self,
        trainer: QuestionAnsweringTrainer,
        train_dataset: Dataset,
    ) -> dict[str, Any]:
        """
        모델 학습을 실행합니다.

        Args:
            trainer (QuestionAnsweringTrainer): 학습을 담당할 Trainer 객체.
            train_dataset (Dataset): 학습에 사용할 데이터셋.

        Returns:
            Any: 학습 결과.
        """
        checkpoint = self.training_args.resume_from_checkpoint or None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        return train_result

    def run_evaluation(
        self,
        trainer: QuestionAnsweringTrainer,
        eval_dataset: Dataset,
    ) -> dict[str, Any]:
        """
        모델 평가를 실행합니다.

        Args:
            trainer (QuestionAnsweringTrainer): 평가를 담당할 Trainer 객체.
            eval_dataset (Dataset): 평가에 사용할 데이터셋.

        Returns:
            Dict[str, Any]: 평가 결과 메트릭.
        """
        return trainer.evaluate(eval_dataset=eval_dataset)
        # fix: eval_example이 인자로 전달되지 않음

    def run_prediction(
        self,
        trainer: QuestionAnsweringTrainer,
        test_dataset: Dataset,
    ) -> BatchEncoding:
        """
        모델 예측을 실행합니다.

        Args:
            trainer (QuestionAnsweringTrainer): 예측을 담당할 Trainer 객체.
            test_dataset (Dataset): 예측에 사용할 데이터셋.

        Returns:
            Any: 예측 결과.
        """
        return trainer.predict(test_dataset=test_dataset)
