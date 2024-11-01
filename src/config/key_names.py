from __future__ import annotations
# NOTE: 딕셔너리의 키, 데이터프레임의 칼럼 명을 관리하는 파일

# (데이터셋)딕셔너리에서 사용되는 상수
ID = 'id'
TRAIN = 'train'
VALIDATION = 'validation'
PREDICTION_TEXT = 'prediction_text'
EXAMPLE_ID = 'example_id'
CONTEXT = 'context'
OFFSET_MAPPING = 'offset_mapping'
SCORE = 'score'
START_LOGIT = 'start_logit'
END_LOGIT = 'end_logit'
TEXT = 'text'
PROBABILITY = 'probability'
NULL_SCORE_DIFF = 'null_odds'
PREDICTIONS = 'predictions'
NBEST_PREDICTIONS = 'nbest_predictions'
PREFIX = 'prefix'

# Retrieve 결과 DataFrame에서 사용되는 상수
ANSWERS = 'answers'
ORIGINAL_CONTEXT = 'original_context'
QUESTION = 'question'
RETRIEVAL_CONTEXT = 'retrieval_context'
UNNAMED = 'Unnamed: 0'
REMOVE_COLUMNS_FROM_RETRIEVER = ['Unnamed: 0', 'original_context']

# DataHandler에서 사용되는 상수
DATA_PROCESSOR = 'DataProc'
PREPROCESSOR = 'pre'
POSTPROCESSOR = 'post'

# reader 토크나이저 혹은 토큰화된 배치(Batch)에서 사용되는 상수
READER_SAMPLE_MAPPING = 'overflow_to_sample_mapping'
READER_OFFSET_MAPPING = 'offset_mapping'
MODEL_INFERED_START = 'start_positions'
MODEL_INFERED_END = 'end_positions'
TOKEN_INPUT_IDS = 'input_ids'
ANSWER_START = 'answer_start'
ANSWER_TEXT = 'text'

# reader trainer에서 사용되는 상수
EVALUATION_DESCRIPTION = 'Evaluation'
PREDICTION_DESCRIPTION = 'Prediction'
