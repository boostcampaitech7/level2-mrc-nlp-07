from __future__ import annotations

import os

from bertviz import head_view
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, utils


class AttentionVisualizer:
    """Attention 시각화를 위한 클래스."""
    def __init__(self, model_name: str):
        utils.logging.set_verbosity_error()  # 경고 메시지 억제
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name, output_attentions=True)

    def encode_input(self, question: str, paragraph: str):
        """
        질문과 단락을 토큰화하여 BERT가 기대하는 형식으로 인코딩합니다.
        """
        inputs = self.tokenizer.encode_plus(
            question,
            paragraph,
            add_special_tokens=True,  # [CLS]와 [SEP] 추가
            return_tensors='pt',      # PyTorch 텐서 반환
        )
        return inputs

    def get_attention(self, inputs):
        """
        모델에 입력을 전달하여 attention 가중치를 얻습니다.
        """
        outputs = self.model(**inputs)
        return outputs[-1]  # attention 가중치는 출력의 마지막 요소에 있습니다.


class HeadViewGenerator:
    """Attention 헤드 뷰를 생성하는 클래스."""
    def __init__(self, output_path: str):
        self.output_path = output_path

    def generate_head_view(self, attention, tokens):
        """
        BERTViz의 head_view를 사용하여 attention heads의 시각적 표현을 생성합니다.
        """
        html_head_view = head_view(attention, tokens, html_action='return')
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w') as file:
            file.write(html_head_view.data)
        print(f'Head view saved to {self.output_path}')


class BERTMRCMonitor:
    """BERT 모델을 사용하여 MRC를 모니터링하는 클래스."""
    def __init__(self, model_name='kykim/bert-kor-base'):
        self.attention_visualizer = AttentionVisualizer(model_name)
        self.head_view_generator = HeadViewGenerator(output_path='./head_view.html')

    def monitor(self, question: str, paragraph: str):
        """
        질문과 단락을 처리하고 attention을 얻고 헤드 뷰를 생성하는 메인 메서드입니다.
        """
        if not question or not paragraph:
            raise ValueError("질문과 단락은 비어있을 수 없습니다.")

        inputs = self.attention_visualizer.encode_input(question, paragraph)
        attention = self.attention_visualizer.get_attention(inputs)
        tokens = self.attention_visualizer.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        self.head_view_generator.generate_head_view(attention, tokens)


# MonitoringTool 사용 예제
if __name__ == '__main__':
    question = '고양이는 무엇을 했나요?'
    paragraph = """
    고양이는 매트 위에 앉아 있었습니다. 날씨는 화창했고, 고양이는 햇살의 따뜻함을 즐겼습니다.
    한편, 개는 그늘에서 쉬고 있었고, 고양이를 호기심 가득한 눈으로 지켜보고 있었습니다.
    """

    monitor = BERTMRCMonitor()
    monitor.monitor(question, paragraph)
