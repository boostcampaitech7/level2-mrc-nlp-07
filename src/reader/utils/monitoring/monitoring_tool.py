from torch.utils.tensorboard import SummaryWriter
import psutil
import GPUtil
import time
import threading
import pandas as pd
from transformers import pipeline
import os
import numpy as np

class MonitoringTool:
    def __init__(self, output_dir: str, monitoring_interval: int = 5):
        os.makedirs(output_dir, exist_ok=True)  # 로그 디렉토리 생성
        self.output_dir = output_dir
        self.writer = SummaryWriter(log_dir=output_dir)
        self.monitoring_thread = None
        self.keep_monitoring = True
        self.data = []
        self.lock = threading.Lock()
        self.monitoring_interval = monitoring_interval

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag: str, values, step: int):
        # 리스트를 numpy 배열로 변환
        values_np = np.array(values)
        self.writer.add_histogram(tag, values_np, step)

    def close(self):
        self.writer.close()

    def start_monitoring(self):
        print("Monitoring started.")
        self.keep_monitoring = True
        self.monitoring_thread = threading.Thread(target=self.monitor_resources)
        self.monitoring_thread.start()

    def stop_monitoring(self):
        print("Stopping monitoring...")
        self.keep_monitoring = False
        self.monitoring_thread.join()

    def monitor_resources(self):
        step = 0

        while self.keep_monitoring:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()

            try:
                gpus = GPUtil.getGPUs()
            except Exception as e:
                print(f"Error accessing GPU information: {e}")
                gpus = []

            # Scalar logging
            self.log_scalar('cpu_usage', cpu_usage, step)
            self.log_scalar('memory_usage', memory_info.percent, step)

            # Histogram logging for CPU and memory
            self.log_histogram('cpu_usage_hist', [cpu_usage], step)
            self.log_histogram('memory_usage_hist', [memory_info.percent], step)

            gpu_data = {}
            for gpu in gpus:
                gpu_usage = gpu.load * 100
                gpu_memory_percent = gpu.memoryUtil * 100
                self.log_scalar(f'gpu_{gpu.id}_usage', gpu_usage, step)
                self.log_scalar(f'gpu_{gpu.id}_memory_usage', gpu_memory_percent, step)

                # GPU histogram logging
                self.log_histogram(f'gpu_{gpu.id}_usage_hist', [gpu_usage], step)
                self.log_histogram(f'gpu_{gpu.id}_memory_usage_hist', [gpu_memory_percent], step)

                gpu_data[f'gpu_{gpu.id}_usage'] = gpu_usage
                gpu_data[f'gpu_{gpu.id}_memory_usage'] = gpu_memory_percent

                print(f"GPU {gpu.id}: Usage: {gpu_usage:.1f}% | Memory Usage: {gpu_memory_percent:.1f}%")

            with self.lock:
                self.data.append({
                    'step': step,
                    'cpu_usage': cpu_usage,
                    'memory_usage': memory_info.percent,
                    **gpu_data
                })

            print(f"CPU Usage: {cpu_usage}% | Memory Usage: {memory_info.percent}% "
                  f"({memory_info.used / (1024 ** 2):.1f}MB / {memory_info.total / (1024 ** 2):.1f}MB)")

            step += 1
            time.sleep(self.monitoring_interval)

        self.save_data()

    def save_data(self):
        """모니터링 데이터를 텍스트 파일로 저장합니다."""
        with open(f'{self.output_dir}/monitoring_data.txt', 'w') as f:
            for entry in self.data:
                line = ', '.join([f"{key}: {value}" for key, value in entry.items()])  # 각 항목을 문자열로 변환
                f.write(line + '\n')  # 각 항목을 한 줄에 기록
        print(f"Monitoring data saved to '{self.output_dir}/monitoring_data.txt'.")


if __name__ == "__main__":
    def monitor_qa(pipeline, context, question):
        result = pipeline(question=question, context=context)
        print(f"Answer: {result['answer']}, Score: {result['score']}")

    monitoring_tool = MonitoringTool(output_dir='logs')
    qa_pipeline = pipeline("question-answering", model="klue/bert-base", device=0)  # 모델 이름 변경
    monitoring_tool.start_monitoring()

    context = "허깅페이스는 자연어 처리(NLP)를 위한 오픈소스 라이브러리를 제공합니다."
    question = "허깅페이스는 무엇을 제공하나요?"
    monitor_qa(qa_pipeline, context, question)

    time.sleep(5)
    monitoring_tool.stop_monitoring()
    monitoring_tool.close()
    print("Monitoring tool closed.")



# TERMINAL : tensorboard --logdir=logs
