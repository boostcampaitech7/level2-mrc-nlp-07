from torch.utils.tensorboard import SummaryWriter
import psutil
import GPUtil
import time
from transformers import pipeline


class MonitoringTool:
    def __init__(self, output_dir: str):
        self.writer = SummaryWriter(log_dir=output_dir)  # TensorBoard Writer 초기화
        self.qa_pipeline = pipeline("question-answering")

    def log_scalar(self, tag: str, value: float, step: int):
        """스칼라 값을 로그에 기록합니다."""
        self.writer.add_scalar(tag, value, step)

    def close(self):
        """TensorBoard Writer를 종료합니다."""
        self.writer.close()

    def start_monitoring(self, duration: int = 60):
        """모니터링 시작 및 종료 시간 설정"""
        print("Monitoring started.")
        end_time = time.time() + duration  # duration 초 후에 종료
        self.monitor_resources(end_time)

    def monitor_resources(self, end_time):
        try:
            step = 0  # 스텝 변수 초기화
            while time.time() < end_time:
                cpu_usage = psutil.cpu_percent(interval=1)
                memory_info = psutil.virtual_memory()
                gpus = GPUtil.getGPUs()

                # CPU 사용량 로그 기록
                self.log_scalar('cpu_usage', cpu_usage, step)
                # 메모리 사용량 로그 기록
                self.log_scalar('memory_usage', memory_info.percent, step)

                for gpu in gpus:
                    gpu_usage = gpu.load * 100
                    gpu_memory_total = gpu.memoryTotal
                    gpu_memory_used = gpu.memoryUsed
                    gpu_memory_percent = gpu.memoryUtil * 100

                    # GPU 사용량 로그 기록
                    self.log_scalar(f'gpu_{gpu.id}_usage', gpu_usage, step)
                    self.log_scalar(f'gpu_{gpu.id}_memory_usage', gpu_memory_percent, step)

                    print(f"GPU {gpu.id}: Usage: {gpu_usage:.1f}% | Memory Usage: {gpu_memory_percent:.1f}% "
                        f"({gpu_memory_used}MB / {gpu_memory_total}MB)")

                print(f"CPU Usage: {cpu_usage}% | Memory Usage: {memory_info.percent}% "
                    f"({memory_info.used / (1024 ** 2):.1f}MB / {memory_info.total / (1024 ** 2):.1f}MB)")

                step += 1  # 스텝 증가
                time.sleep(5)
        except Exception as e:
            print(f"Error during monitoring: {e}")

    def monitor_qa(self, context, question):
        result = self.qa_pipeline(question=question, context=context)
        print(f"Answer: {result['answer']}, Score: {result['score']}")


# MonitoringTool 사용 예제
if __name__ == "__main__":
    # MonitoringTool 인스턴스 생성
    monitoring_tool = MonitoringTool(output_dir='logs')

    # 모니터링 시작 (10초 동안)
    try:
        monitoring_tool.start_monitoring(duration=10)

        # 질문-답변 기능 테스트
        context = "허깅페이스는 자연어 처리(NLP)를 위한 오픈소스 라이브러리를 제공합니다."
        question = "허깅페이스는 무엇을 제공하나요?"
        monitoring_tool.monitor_qa(context, question)

    except KeyboardInterrupt:
        print("Monitoring interrupted by user.")
    finally:
        # TensorBoard Writer 종료
        monitoring_tool.close()
        print("Monitoring tool closed.")


# terminal 입력: tensorboard --logdir=logs
