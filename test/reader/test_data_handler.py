from __future__ import annotations

import sys
import unittest

from datasets import Dataset
from reader.data_handler import DataHandler
from reader.utils.arguments import DataTrainingArguments
from transformers import AutoTokenizer
sys.path.append('/data/ephemeral/home/level2-mrc-nlp-07/src')


class TestDataHandler(unittest.TestCase):
    def setUp(self) -> None:
        # 테스트를 위한 DataTrainingArguments 인스턴스를 생성합니다.
        self.data_args = DataTrainingArguments()  # 필요한 초기화 매개변수를 제공하십시오.

        # 사용할 tokenizer 초기화 (예시: 'bert-base-uncased' 사용)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        # DataHandler 인스턴스 생성
        self.data_handler = DataHandler(self.data_args, self.tokenizer)

    def test_load_data(self) -> None:
        # load_data 메서드를 호출하여 데이터가 로드되는지 테스트합니다.
        result = self.data_handler.load_data(type='train')

        # 반환된 값이 Dataset 인스턴스인지 확인합니다.
        self.assertTrue(
            isinstance(result, Dataset),  # Dataset 클래스인지 확인
            'load_data should return an instance of Dataset',
        )

    def test_process_data(self) -> None:
        # process_data 메서드를 호출하여 데이터가 처리되는지 테스트합니다.
        result = self.data_handler.process_data()
        self.assertEqual(
            result, {'data': 'processed'},
            "process_data should return {'data': 'processed'}",
        )


if __name__ == '__main__':
    unittest.main()
