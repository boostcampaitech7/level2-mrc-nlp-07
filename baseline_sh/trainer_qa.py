from transformers import is_datasets_available
# from transformers import is_torch_tpu_available
from transformers import Trainer
from typing import Optional, Callable, Any, Dict

if is_datasets_available():
    import datasets

# if is_torch_tpu_available():
#     import torch_xla.core.xla_model as xm
#     import torch_xla.debug.metrics as met


class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args: Any, eval_examples: Optional[list] = None, post_process_function: Optional[Callable] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.eval_examples: Optional[list] = eval_examples
        self.post_process_function: Optional[Callable] = post_process_function

    def _shared_evaluate_or_predict(
        self, 
        dataset: Any, 
        examples: list, 
        description: str, 
        ignore_keys: Optional[list] = None, 
        is_predict: bool = False
    ) -> Any:
        # 데이터로더 선택
        dataloader = (
            self.get_test_dataloader(dataset)
            if is_predict
            else self.get_eval_dataloader(dataset)
        )

        # 메트릭 계산 비활성화
        compute_metrics = self.compute_metrics
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

        # 데이터셋 형식 설정
        if isinstance(dataset, datasets.Dataset):
            dataset.set_format(
                type=dataset.format["type"],
                columns=list(dataset.features.keys()),
            )

        # 후처리 및 예측값 반환
        if self.post_process_function is not None:
            return self.post_process_function(examples, dataset, output.predictions, self.args)
        return output

    def evaluate(self, eval_dataset: Optional[Any] = None, eval_examples: Optional[list] = None, ignore_keys: Optional[list] = None) -> Dict[str, Any]:
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # 평가 수행
        eval_preds = self._shared_evaluate_or_predict(
            eval_dataset, eval_examples, description="Evaluation", ignore_keys=ignore_keys
        )

        # 메트릭 계산 및 로깅
        metrics: Dict[str, Any] = {}
        if self.compute_metrics is not None:
            metrics = self.compute_metrics(eval_preds)
            self.log(metrics)

        # # TPU 메트릭 디버그
        # if self.args.tpu_metrics_debug or self.args.debug:
        #     xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        return metrics

    def predict(self, test_dataset: Any, test_examples: list, ignore_keys: Optional[list] = None) -> Any:
        return self._shared_evaluate_or_predict(
            test_dataset, test_examples, description="Prediction", ignore_keys=ignore_keys, is_predict=True
        )
