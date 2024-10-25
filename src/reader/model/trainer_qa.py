from __future__ import annotations

from typing import Any
from typing import Callable

from transformers import BatchEncoding
from transformers import is_datasets_available
from transformers import is_torch_xla_available
from transformers import Trainer

if is_datasets_available():
    import datasets

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


class QuestionAnsweringTrainer(Trainer):
    def __init__(
        self,
        *args,
        eval_examples: datasets.Dataset | None = None,
        post_process_function: Callable | None = None,
        **kwargs,
    ):
        """
        QuestionAnsweringTrainer는 Hugging Face의 Trainer를 확장하여
        후처리 및 평가 예시 기능을 추가한 클래스입니다.

        Args:
            eval_examples (Optional[datasets.Dataset]): 평가를 위한 원본 예시 데이터셋.
            post_process_function (Optional[callable]): 예측 후처리 함수, 원본 예시와 예측값을 처리하는 함수.
        """
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def _shared_evaluate_or_predict(
        self,
        dataset: datasets.Dataset,
        examples: datasets.Dataset,
        description: str,
        ignore_keys: str | list | None = None,
        is_predict: bool = False,
    ) -> dict[str, Any] | Any:
        """
        평가 또는 예측 작업을 수행하는 내부 공통 메서드입니다.

        Args:
            dataset (datasets.Dataset): 평가 또는 예측에 사용할 데이터셋.
            examples (datasets.Dataset): 원본 평가 또는 예측 예시 데이터셋.
            description (str): 평가 또는 예측 작업에 대한 설명.
            ignore_keys (Optional[Union[str, list]]): 로깅에서 무시할 키 값.
            is_predict (bool): 예측 작업인지 여부를 나타내는 플래그.

        Returns:
            Union[Dict[str, Any], Any]: 후처리된 예측값 또는 평가/예측 결과.
        """
        dataloader = (
            self.get_test_dataloader(dataset)
            if is_predict
            else self.get_eval_dataloader(dataset)
        )

        compute_metrics = self.compute_metrics  # type: ignore
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                dataloader,
                description=description,
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if isinstance(dataset, datasets.Dataset):
            dataset.set_format(
                type=dataset.format['type'],
                columns=list(dataset.features.keys()),
            )

        if self.post_process_function is not None:
            preds = self.post_process_function(
                self.args, examples, dataset, output.predictions,
            )
            return preds
        else:
            return output

    def evaluate(
        self,
        eval_dataset: datasets.Dataset | None = None,
        eval_examples: datasets.Dataset | None = None,
        ignore_keys: str | list | None = None,
    ) -> dict[str, Any]:
        """
        평가 작업을 수행하는 메서드입니다.

        Args:
            eval_dataset (Optional[datasets.Dataset]): 평가에 사용할 데이터셋. 기본값은 None이며, 이 경우 `self.eval_dataset`을 사용합니다.
            eval_examples (Optional[datasets.Dataset]): 평가에 사용할 원본 예시 데이터셋.
            기본값은 None이며, 이 경우 `self.eval_examples`을 사용합니다.
            ignore_keys (Optional[Union[str, list]]): 로깅에서 무시할 키 값.

        Returns:
            Dict[str, Any]: 평가 결과 메트릭을 포함한 딕셔너리.
        """
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # 예측 수행 (후처리 및 결과)
        eval_preds = self._shared_evaluate_or_predict(
            eval_dataset, eval_examples, description='Evaluation', ignore_keys=ignore_keys,
        )

        # 메트릭 계산 및 로깅
        if self.compute_metrics is not None:
            metrics = self.compute_metrics(eval_preds)
            self.log(metrics)
        else:
            metrics = {}

        if self.args.tpu_metrics_debug or self.args.debug:
            xm.master_print(met.metrics_report())

        self.control: Any = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics,
        )
        return metrics

    def predict(
        self,
        test_dataset: datasets.Dataset,
        test_examples: datasets.Dataset | None = None,
        ignore_keys: str | list | None = None,
    ) -> BatchEncoding:
        """
        예측 작업을 수행하는 메서드입니다.

        Args:
            test_dataset (datasets.Dataset): 예측에 사용할 데이터셋.
            test_examples (datasets.Dataset): 예측에 사용할 원본 예시 데이터셋.
            ignore_keys (Optional[Union[str, list]]): 로깅에서 무시할 키 값.

        Returns:
            Union[Dict[str, Any], Any]: 후처리된 예측값 또는 예측 결과.
        """
        test_examples = self.eval_examples if test_examples is None else test_examples

        return self._shared_evaluate_or_predict(
            test_dataset, test_examples, description='Prediction', ignore_keys=ignore_keys, is_predict=True,
        )
