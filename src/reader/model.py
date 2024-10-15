import torch
from arguments import ModelArguments
from transformers import AutoConfig
from transformers import AutoModelForQuestionAnswering
from transformers import AutoTokenizer


class Reader:
    # TODO: 리더 클래스 개발
    def __init__(self, model_args: ModelArguments) -> None:
        self.config = AutoConfig.from_pretrained(
            model_args.config_name
            if model_args.config_name is not None
            else model_args.model_name_or_path,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name
            if model_args.tokenizer_name is not None
            else model_args.model_name_or_path,
            use_fast=True,
        )
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool('.ckpt' in model_args.model_name_or_path),
            config=self.config,
        )
        self.output: dict = {}

    def train(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> dict:
        if self.model:
            self.output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        return self.output

    def predict(self, output: dict) -> dict:
        return {'predictions': output}
