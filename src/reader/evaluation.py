class Evaluation:
    def compute_metrics(
        self,
        predictions: list,
        labels: list,
    ) -> dict[str, float]:
        # TODO: 메트릭 계산 로직
        accuracy = 0.9  # 예시
        return {'accuracy': accuracy}
