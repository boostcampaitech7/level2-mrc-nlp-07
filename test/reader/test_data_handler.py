from __future__ import annotations

import sys
sys.path.append('/data/ephemeral/home/level2-mrc-nlp-07/src')

import unittest
from arguments import DataTrainingArguments
from reader.data_handler import DataHandler


class TestDataHandler(unittest.TestCase):
    def setUp(self) -> None:
        # 테스트를 위한 DataTrainingArguments 인스턴스를 생성합니다.
        self.data_args = DataTrainingArguments()  # 필요한 초기화 매개변수를 제공하십시오.
        self.data_handler = DataHandler(self.data_args)

    def test_load_data(self) -> None:
        # load_data 메서드를 호출하여 데이터가 로드되는지 테스트합니다.
        result = self.data_handler.load_data()
        self.assertEqual(
            result, {'data': 'loaded'},
            "load_data should return {'data': 'loaded'}",
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
