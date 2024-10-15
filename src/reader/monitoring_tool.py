from typing import Any


class MonitoringTool:
    # TODO: 모니터링 도구 적용
    def start_monitoring(self) -> None:
        print('Monitoring started.')

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        print(f'Metrics logged: {metrics}')

    def stop_monitoring(self) -> None:
        print('Monitoring stopped.')
