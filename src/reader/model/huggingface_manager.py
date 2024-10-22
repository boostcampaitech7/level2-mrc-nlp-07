from __future__ import annotations

from transformers import AutoConfig
from transformers import AutoModelForQuestionAnswering
from transformers import AutoTokenizer

from utils.arguments import ModelArguments


class HuggingFaceLoadManager:
    def __init__(self, model_args: ModelArguments):
        self.model_args = model_args
        self.config = None
        self.tokenizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        """
        허깅페이스, 혹은 저장된 모델 파일을 불러옵니다.
        """
        try:
            self.config = AutoConfig.from_pretrained(
                self.model_args.config_name or self.model_args.model_name_or_path,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_args.tokenizer_name or self.model_args.model_name_or_path, use_fast=True,
            )
            self.model = AutoModelForQuestionAnswering.from_pretrained(
                self.model_args.model_name_or_path,
                config=self.config,
            )
        except Exception as e:
            raise RuntimeError(f'모델을 로드하는 중 오류 발생: {e}')

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer
