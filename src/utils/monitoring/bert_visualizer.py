from __future__ import annotations

from bertviz import head_view
import os
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, utils


class BERTMRCMonitor:
    def __init__(self, model_name='kykim/bert-kor-base'):
        utils.logging.set_verbosity_error()  # Suppress standard warnings
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name, output_attentions=True)

    def encode_input(self, question: str, paragraph: str):
        """
        Tokenize and encode the question and paragraph into the format expected by BERT.
        """
        inputs = self.tokenizer.encode_plus(
            question,
            paragraph,
            add_special_tokens=True,  # Adds [CLS] and [SEP]
            return_tensors='pt',       # Returns PyTorch tensors
        )
        return inputs

    def get_attention(self, inputs):
        """
        Pass the inputs through the model to obtain attention weights.
        """
        outputs = self.model(**inputs)
        return outputs[-1]  # Attention weights are in the last element when output_attentions=True

    def generate_head_view(self, attention, tokens, output_path):
        """
        Use BERTViz's head_view to generate a visual representation of attention heads.
        """
        html_head_view = head_view(attention, tokens, html_action='return')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as file:
            file.write(html_head_view.data)
        print(f'Head view saved to {output_path}')

    def monitor(self, question, paragraph, output_path='./head_view.html'):
        """
        Main method to process a question and paragraph, get the attention, and generate the head view.
        """
        if not question or not paragraph:
            raise ValueError("질문과 단락은 비어있을 수 없습니다.")
        
        inputs = self.encode_input(question, paragraph)
        attention = self.get_attention(inputs)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        self.generate_head_view(attention, tokens, output_path)


# MonitoringTool 사용 예제
if __name__ == '__main__':
    question = '고양이는 무엇을 했나요?'
    paragraph = """
    고양이는 매트 위에 앉아 있었습니다. 날씨는 화창했고, 고양이는 햇살의 따뜻함을 즐겼습니다.
    한편, 개는 그늘에서 쉬고 있었고, 고양이를 호기심 가득한 눈으로 지켜보고 있었습니다.
    """

    monitor = BERTMRCMonitor()
    monitor.monitor(question, paragraph)
