from __future__ import annotations

import sys
sys.path.append('/data/ephemeral/home/level2-mrc-nlp-07/src')

import unittest
from reader.evaluation import Evaluation


class TestEvaluation(unittest.TestCase):

    def setUp(self) -> None:
        # Evaluation 클래스의 인스턴스를 생성합니다.
        self.evaluation = Evaluation()

    def test_compute_metrics(self) -> None:
        # 예측값과 라벨을 정의합니다.
        predictions = [1, 0, 1, 1, 0]  # 예시 예측
        labels = [1, 0, 1, 0, 0]        # 예시 실제 라벨

        # compute_metrics 메서드를 호출하여 메트릭을 계산합니다.
        result = self.evaluation.compute_metrics(predictions, labels)

        # 예상되는 정확도 계산 (여기서는 3/5 = 0.6)
        expected_accuracy = 0.6

        # 정확도를 비교합니다.
        self.assertAlmostEqual(
            result['accuracy'], expected_accuracy, places=2,
            msg='Computed accuracy should match the expected value.',
        )


if __name__ == '__main__':
    unittest.main()
