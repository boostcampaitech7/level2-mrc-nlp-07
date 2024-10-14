
# 모니터링 시작 메시지가 정상적으로 출력되는지 확인
def test_start_monitoring(capfd):
    monitor_tool = MonitoringTool()
    monitor_tool.start_monitoring()
    captured = capfd.readouterr()
    assert 'Monitoring started.' in captured.out


# 로그된 메트릭이 올바르게 출력되는지 확인
def test_log_metrics(capfd):
    monitor_tool = MonitoringTool()
    metrics = {'accuracy': 0.9}
    monitor_tool.log_metrics(metrics)
    captured = capfd.readouterr()
    assert 'Metrics logged: {\'accuracy\': 0.9}' in captured.out
