from __future__ import annotations

import pytest
from transformers import EvalPrediction

from src import DataPostProcessor  # 여기에 DataPostProcessor의 경로를 적어주세요
from src import DataTrainingArguments  # DataTrainingArguments의 경로를 적어주세요
from src.reader.data_controller.postprocess_qa import postprocess_qa_predictions  # postprocess_qa_predictions의 경로를 적어주세요


@pytest.fixture
def sample_data():
    # 테스트 데이터 생성
    return {
        'title': '2011년 함부르크 주의회 선거',
        'context': ('2008년 2월 28일 실시된 2008년 함부르크 주의회 선거에서 기민련은 '
                    '과반수 의석에 못 미치는 다수당이 되었다. 기민련은 녹색당과 함께 '
                    '올레 폰 보이스트를 시장으로 하는 흑록연정을 구성했다. 이는 독일연방과 '
                    '주에서 시도된 최초의 기민련과 녹색당간의 연정이었다.\n\n연정 구성당시의 '
                    '시장이 은퇴선언으로 물러나고, 크리스토포 알하우스가 시장으로 취임한 후 '
                    '기민련-녹색당 연정이 갈등을 빚어왔다. 2010년 11월 28일 녹색당이 연정탈퇴와 '
                    '조기 선거지지를 선언했다. 그 이후 의회내 다수의 지지가 없는 기민련-단독 '
                    '정부가 함부르크 주정부를 이끌었다.\n\n사민당 시장 후보인 올라프 숄츠는 '
                    '기자회견에서 자신 주도하의 적녹연정을 지지한다고 밝힌바 있다.\n\n여러 '
                    '여론 조사 결과에 의하면 지난 선거에 비해 사민당과 녹색당의 지지율이 높게 '
                    '나오고 있으며, 기민련의 경우 30% 이하의 지지율을 보이고 있다. 좌파당과 '
                    '자민당의 경우 5% 봉쇄조항을 약간 상회하는 지지율을 보이고 있다. 해적당을 '
                    '비롯한 기타 정당의 경우 봉쇄조항 이상의 득표는 어려울 것으로 조사됐다.'),
        'question': '올레 폰 보이스트 이후에 시장으로 임명된 사람은 누구인가?',
        'id': 'mrc-0-003589',
        'answers': {'answer_start': [189], 'text': ['크리스토포 알하우스']},
        'document_id': 27803,
        '__index_level_0__': 2354
    }


def test_data_post_processor_predict(sample_data):
    # DataPostProcessor 인스턴스를 생성하고 process 메서드를 호출합니다.
    tokenizer = None  # 필요한 경우 실제 토크나이저를 여기에 정의하세요.
    data_args = DataTrainingArguments(max_answer_length=30)  # DataTrainingArguments 인스턴스 생성
    examples = [sample_data]
    features = [{'input_ids': [0], 'attention_mask': [1], 'id': sample_data['id']}]  # 예시 특성
    predictions = [{'start_logits': [0.1], 'end_logits': [0.2]}]  # 예시 예측 결과

    # process 메서드를 호출하여 예측 결과를 얻습니다.
    formatted_predictions = DataPostProcessor.process('predict', tokenizer, data_args, examples, features, predictions)

    # 반환된 형식화된 예측 결과를 검증합니다.
    assert isinstance(formatted_predictions, list)
    assert len(formatted_predictions) == 1
    assert formatted_predictions[0]['id'] == sample_data['id']
    assert formatted_predictions[0]['prediction_text'] == '크리스토포 알하우스'  # 예상되는 결과


def test_data_post_processor_eval(sample_data):
    # DataPostProcessor 인스턴스를 생성하고 process 메서드를 호출합니다.
    tokenizer = None  # 필요한 경우 실제 토크나이저를 여기에 정의하세요.
    data_args = DataTrainingArguments(max_answer_length=30)  # DataTrainingArguments 인스턴스 생성
    examples = [sample_data]
    features = [{'input_ids': [0], 'attention_mask': [1], 'id': sample_data['id']}]  # 예시 특성
    predictions = [{'start_logits': [0.1], 'end_logits': [0.2]}]  # 예시 예측 결과

    # process 메서드를 호출하여 평가 결과를 얻습니다.
    eval_prediction = DataPostProcessor.process('eval', tokenizer, data_args, examples, features, predictions)

    # 반환된 EvalPrediction 객체를 검증합니다.
    assert isinstance(eval_prediction, EvalPrediction)
    assert len(eval_prediction.predictions) == 1
    assert eval_prediction.predictions[0]['id'] == sample_data['id']
    assert eval_prediction.predictions[0]['prediction_text'] == '크리스토포 알하우스'  # 예상되는 결과
