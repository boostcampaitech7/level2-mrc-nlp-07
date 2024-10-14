
# 주어진 predictions와 labels로 정확한 메트릭을 계산하는지 확인
def test_compute_metrics():
    evaluation = Evaluation()
    predictions = [1, 0, 1, 1]
    labels = [1, 0, 0, 1]
    metrics = evaluation.compute_metrics(predictions, labels)
    assert 'accuracy' in metrics
    assert metrics['accuracy'] == 0.75
