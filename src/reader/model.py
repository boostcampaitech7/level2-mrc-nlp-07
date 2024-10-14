from __future__ import annotations

import torch
from arguments import ModelArguments
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer


class Reader:
    # TODO: 리더 클래스 개발
    def __init__(self, model_args: ModelArguments) -> None:
        self.model_name_or_path: str = model_args.model_name_or_path
        self.model: PreTrainedModel | None = None
        self.tokenizer: PreTrainedTokenizer | None = None
        self.config: PretrainedConfig | None = None
        self.output: dict = {}

    def load_model(self) -> None:
        self.model = PreTrainedModel.from_pretrained(self.model_name_or_path)
        self.config = PretrainedConfig.from_pretrained(self.model_name_or_path)

    def load_tokenizer(self) -> None:
        self.tokenizer = PreTrainedTokenizer.from_pretrained(
            self.model_name_or_path,
        )

    def forward(
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

    def post_process(self, output: dict) -> dict:
        return {'predictions': output}
