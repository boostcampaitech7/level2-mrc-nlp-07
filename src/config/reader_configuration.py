from __future__ import annotations
# NOTE: Reader 단에서 사용되는 설정 값(하이퍼파라미터 혹은 기타 옵션)을 관리하는 모듈

# 토크나이저 설정
TOKENIZER_PADDING_DIRECTION = 'right'
PADDING_OPTION = 'max_length'
TOKENIZER_TRUNCATE_FIRST = 'only_first'
TOKENIZER_TRUNCATE_SECOND = 'only_second'

# 후처리 설정
N_BEST_SIZE = 20
MAX_ANSWER_LENGTH = 30
NULL_SCORE_DIFF_THRESHOLD = 0.0
