from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict

import GPUtil
import numpy as np
import psutil
from torch.utils.tensorboard import SummaryWriter
from transformers import pipeline


class ResourceMonitor:
    """시스템 리소스를 모니터링하는 클래스."""
    def __init__(self, monitoring_interval: int = 5):
        self.monitoring_interval = monitoring_interval
        self.data: list[Dict[str, Any]] = []  # 모니터링 데이터를 저장할 리스트
        self.lock = threading.Lock()

    def collect_data(self) -> Dict[str, Any]:
        """CPU, 메모리 및 GPU 데이터를 수집합니다."""
        cpu_usage = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        gpus = self._get_gpu_data()

        # 수집한 데이터를 딕셔너리 형태로 반환
        return {
            'cpu_usage': cpu_usage,
            'memory_usage': memory_info.percent,
            **gpus
        }

    def _get_gpu_data(self) -> Dict[str, float]:
        """GPU 정보를 수집합니다."""
        gpu_data = {}
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_usage = gpu.load * 100
                gpu_memory_percent = gpu.memoryUtil * 100
                gpu_data[f'gpu_{gpu.id}_usage'] = gpu_usage
                gpu_data[f'gpu_{gpu.id}_memory_usage'] = gpu_memory_percent
        except Exception as e:
            print(f'Error accessing GPU information: {e}')
        return gpu_data


class Logger:
    """TensorBoard에 로그를 기록하는 클래스."""
    def __init__(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)  # 로그 디렉토리 생성
        self.writer = SummaryWriter(log_dir=output_dir)

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag: str, values, step: int):
        values_np = np.array(values)
        self.writer.add_histogram(tag, values_np, step)

    def close(self):
        self.writer.close()


class MonitoringTool:
    """리소스 모니터링을 위한 클래스."""
    def __init__(self, output_dir: str, monitoring_interval: int = 5):
        self.logger = Logger(output_dir)
        self.monitor = ResourceMonitor(monitoring_interval)
        self.monitoring_thread = None
        self.keep_monitoring = True

    def start_monitoring(self):
        print('Monitoring started.')
        self.keep_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor_resources)
        self.monitoring_thread.start()

    def stop_monitoring(self):
        print('Stopping monitoring...')
        self.keep_monitoring = False
        self.monitoring_thread.join()
        self.save_data()

    def _monitor_resources(self):
        step = 0

        while self.keep_monitoring:
            resource_data = self.monitor.collect_data()

            # 로그 기록
            self._log_resources(resource_data, step)

            step += 1
            time.sleep(self.monitor.monitoring_interval)

    def _log_resources(self, resource_data: Dict[str, Any], step: int):
        """리소스 데이터를 로그에 기록합니다."""
        # Scalar logging
        self.logger.log_scalar('cpu_usage', resource_data['cpu_usage'], step)
        self.logger.log_scalar('memory_usage', resource_data['memory_usage'], step)

        # Histogram logging for CPU and memory
        self.logger.log_histogram('cpu_usage_hist', [resource_data['cpu_usage']], step)
        self.logger.log_histogram('memory_usage_hist', [resource_data['memory_usage']], step)

        for key, value in resource_data.items():
            if key.startswith('gpu_'):
                self.logger.log_scalar(key, value, step)
                self.logger.log_histogram(f'{key}_hist', [value], step)

        print(self._format_resource_output(resource_data))

    def _format_resource_output(self, resource_data: Dict[str, Any]) -> str:
        """리소스 데이터를 출력 형식으로 포맷합니다."""
        output_lines = [f"{key}: {value:.1f}%" for key, value in resource_data.items()]
        return ' | '.join(output_lines)

    def save_data(self):
        """모니터링 데이터를 텍스트 파일로 저장합니다."""
        with open(f'{self.logger.writer.log_dir}/monitoring_data.txt', 'w') as f:
            for entry in self.monitor.data:
                line = ', '.join([f'{key}: {value}' for key, value in entry.items()])
                f.write(line + '\n')
        print(f"Monitoring data saved to '{self.logger.writer.log_dir}/monitoring_data.txt'.")

    def close(self):
        self.logger.close()


class QAExecutor:
    """질문 답변 기능을 실행하는 클래스."""
    def __init__(self, model_name: str, device: int):
        self.pipeline = pipeline('question-answering', model=model_name, device=device)

    def ask_question(self, context: str, question: str):
        result = self.pipeline(question=question, context=context)
        print(f"Answer: {result['answer']}, Score: {result['score']}")


if __name__ == '__main__':
    monitoring_tool = MonitoringTool(output_dir='logs')
    qa_executor = QAExecutor(model_name='klue/bert-base', device=0)

    monitoring_tool.start_monitoring()

    context = '허깅페이스는 자연어 처리(NLP)를 위한 오픈소스 라이브러리를 제공합니다.'
    question = '허깅페이스는 무엇을 제공하나요?'
    qa_executor.ask_question(context, question)

    time.sleep(5)
    monitoring_tool.stop_monitoring()
    monitoring_tool.close()
    print('Monitoring tool closed.')
