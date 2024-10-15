import unittest
from unittest.mock import patch

from ..monitoring_tool import MonitoringTool


class TestMonitoringTool(unittest.TestCase):

    @patch('builtins.print')  # print 함수의 동작을 mock 처리합니다.
    def test_start_monitoring(self, mock_print):
        tool = MonitoringTool()
        tool.start_monitoring()
        mock_print.assert_called_once_with('Monitoring started.')  # 예상 출력 확인

    @patch('builtins.print')
    def test_log_metrics(self, mock_print):
        tool = MonitoringTool()
        metrics = {'accuracy': 0.95, 'loss': 0.05}
        tool.log_metrics(metrics)
        mock_print.assert_called_once_with(
            f'Metrics logged: {metrics}',
        )  # 예상 출력 확인

    @patch('builtins.print')
    def test_stop_monitoring(self, mock_print):
        tool = MonitoringTool()
        tool.stop_monitoring()
        mock_print.assert_called_once_with('Monitoring stopped.')  # 예상 출력 확인


if __name__ == '__main__':
    unittest.main()
