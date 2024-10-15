import os
import sys
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import torch

sys.path.append('/data/ephemeral/home/level2-mrc-nlp-07/src')
from reader.model import Reader
from arguments import ModelArguments


class TestReader(unittest.TestCase):

    def setUp(self) -> None:
        # ModelArguments의 mock 객체를 생성합니다.
        self.model_args = ModelArguments(model_name_or_path='mock_model')
        self.reader = Reader(model_args=self.model_args)

    @patch('transformers.PreTrainedModel.from_pretrained')
    @patch('transformers.PretrainedConfig.from_pretrained')
    def test_load_model(self, mock_config, mock_model):
        # Mock 객체의 return_value 설정
        mock_model.return_value = MagicMock()
        mock_config.return_value = MagicMock()

        # load_model 호출
        self.reader.load_model()

        # 모델과 설정이 로드되었는지 확인
        self.assertIsNotNone(self.reader.model)
        self.assertIsNotNone(self.reader.config)
        mock_model.assert_called_once_with(self.model_args.model_name_or_path)
        mock_config.assert_called_once_with(self.model_args.model_name_or_path)

    @patch('transformers.PreTrainedTokenizer.from_pretrained')
    def test_load_tokenizer(self, mock_tokenizer):
        # Mock 객체의 return_value 설정
        mock_tokenizer.return_value = MagicMock()

        # load_tokenizer 호출
        self.reader.load_tokenizer()

        # 토크나이저가 로드되었는지 확인
        self.assertIsNotNone(self.reader.tokenizer)
        mock_tokenizer.assert_called_once_with(
            self.model_args.model_name_or_path,
        )

    def test_forward(self):
        # mock 입력값
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])
        token_type_ids = torch.tensor([[0, 0, 0]])

        # 모델이 None이 아닌 경우의 동작 테스트
        self.reader.model = MagicMock()
        self.reader.model.return_value = {'logits': torch.tensor([[0.1, 0.9]])}

        output = self.reader.forward(input_ids, attention_mask, token_type_ids)

        # output이 모델의 출력과 같은지 확인
        self.assertEqual(output, {'logits': torch.tensor([[0.1, 0.9]])})

    def test_post_process(self):
        output = {'logits': torch.tensor([[0.1, 0.9]])}

        result = self.reader.post_process(output)

        # post_process의 결과가 예상한 형식인지 확인
        self.assertEqual(result, {'predictions': output})


if __name__ == '__main__':
    unittest.main()
