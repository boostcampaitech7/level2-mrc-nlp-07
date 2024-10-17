from transformers import is_datasets_available
from transformers import is_torch_tpu_available
from transformers import Trainer

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def _shared_evaluate_or_predict(
        self, dataset, examples, description, ignore_keys=None, is_predict=False
    ):
        dataloader = (
            self.get_test_dataloader(dataset)
            if is_predict
            else self.get_eval_dataloader(dataset)
        )

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

        if isinstance(dataset, datasets.Dataset):
            dataset.set_format(
                type=dataset.format["type"],
                columns=list(dataset.features.keys()),
            )

        if self.post_process_function is not None:
            preds = self.post_process_function(
                examples, dataset, output.predictions, self.args
            )
            return preds
        else:
            return output

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # 예측 수행 (후처리 및 결과)
        eval_preds = self._shared_evaluate_or_predict(
            eval_dataset, eval_examples, description="Evaluation", ignore_keys=ignore_keys
        )

        # 메트릭 계산 및 로깅
        if self.compute_metrics is not None:
            metrics = self.compute_metrics(eval_preds)
            self.log(metrics)
        else:
            metrics = {}

        if self.args.tpu_metrics_debug or self.args.debug:
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        return metrics

    def predict(self, test_dataset, test_examples, ignore_keys=None):
        return self._shared_evaluate_or_predict(
            test_dataset, test_examples, description="Prediction", ignore_keys=ignore_keys, is_predict=True
        )
