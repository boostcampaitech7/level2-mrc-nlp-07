from typing import Any
from typing import Callable
from typing import Optional
from typing import Union

from datasets import Dataset
from transformers import is_torch_tpu_available
from transformers import Trainer
from transformers.trainer_utils import EvalPrediction


if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


class QuestionAnsweringTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        eval_examples: Optional[Dataset] = None,
        post_process_function: Callable | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.eval_examples: Optional[Dataset] = eval_examples
        self.post_process_function: Callable | None = post_process_function

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        eval_examples: Optional[Dataset] = None,
        ignore_keys: Optional[Union[str, list]] = None,
    ) -> dict[str, float]:
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        compute_metrics = self.compute_metrics  # type: ignore
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                eval_dataloader,
                description='Evaluation',
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if isinstance(eval_dataset, Dataset):
            eval_dataset.set_format(
                type=eval_dataset.format['type'],
                columns=list(eval_dataset.features.keys()),
            )

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds: EvalPrediction = self.post_process_function(
                eval_examples, eval_dataset, output.predictions, self.args,
            )
            metrics: dict[str, float] = self.compute_metrics(eval_preds)
            self.log(metrics)
        else:
            metrics = {}

        if self.args.tpu_metrics_debug or self.args.debug:
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics,   # type: ignore
        )
        return metrics

    def predict(
        self,
        test_dataset: Dataset,
        test_examples: Dataset,
        ignore_keys: Optional[Union[str, list]] = None,
    ) -> Any:
        test_dataloader = self.get_test_dataloader(test_dataset)

        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                test_dataloader,
                description='Evaluation',
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        if isinstance(test_dataset, Dataset):
            test_dataset.set_format(
                type=test_dataset.format['type'],
                columns=list(test_dataset.features.keys()),
            )

        predictions = self.post_process_function(
            test_examples, test_dataset, output.predictions, self.args,
        )
        return predictions
