from __future__ import annotations

import collections
import json
import os

import numpy as np
from tqdm.auto import tqdm

from src.utils.constants.key_names import CONTEXT
from src.utils.constants.key_names import END_LOGIT
from src.utils.constants.key_names import EXAMPLE_ID
from src.utils.constants.key_names import ID
from src.utils.constants.key_names import NBEST_PREDICTIONS
from src.utils.constants.key_names import NULL_SCORE_DIFF
from src.utils.constants.key_names import OFFSET_MAPPING
from src.utils.constants.key_names import PREDICTIONS
from src.utils.constants.key_names import PROBABILITY
from src.utils.constants.key_names import SCORE
from src.utils.constants.key_names import START_LOGIT
from src.utils.constants.key_names import TEXT
from src.utils.constants.reader_configuration import MAX_ANSWER_LENGTH
from src.utils.constants.reader_configuration import N_BEST_SIZE
from src.utils.constants.reader_configuration import NULL_SCORE_DIFF_THRESHOLD
from src.utils.log.logger import setup_logger


def postprocess_qa_predictions(
    examples,
    features,
    predictions: tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = N_BEST_SIZE,
    max_answer_length: int = MAX_ANSWER_LENGTH,
    null_score_diff_threshold: float = NULL_SCORE_DIFF_THRESHOLD,
    output_dir: str | None = None,
    prefix: str | None = None,
    is_world_process_zero: bool = True,
):
    """
    Post-processes : qa model의 prediction 값을 후처리하는 함수
    모델은 start logit과 end logit을 반환하기 때문에, 이를 기반으로 original text로 변경하는 후처리가 필요함

    Args:
        examples: 전처리 되지 않은 데이터셋 (see the main script for more information).
        features: 전처리가 진행된 데이터셋 (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            모델의 예측값 :start logits과 the end logits을 나타내는 two arrays
            첫번째 차원은 :obj:`features`의 element와 갯수가 맞아야함.
        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
            정답이 없는 데이터셋이 포함되어있는지 여부를 나타냄
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            답변을 찾을 때 생성할 n-best prediction 총 개수
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            생성할 수 있는 답변의 최대 길이
        null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
            null 답변을 선택하는 데 사용되는 threshold
            : if the best answer has a score that is less than the score of
            the null answer minus this threshold, the null answer is selected for this example (note that the score of
            the null answer for an example giving several features is the minimum of the scores for the null answer on
            each feature: all features must be aligned on the fact they `want` to predict a null answer).
            Only useful when :obj:`version_2_with_negative` is :obj:`True`.
        output_dir (:obj:`str`, `optional`):
            아래의 값이 저장되는 경로
            dictionary : predictions, n_best predictions (with their scores and logits)
                if:obj:`version_2_with_negative=True`,
            dictionary : the scores differences between best and null answers
        prefix (:obj:`str`, `optional`):
            dictionary에 `prefix`가 포함되어 저장됨
        is_world_process_zero (:obj:`bool`, `optional`, defaults to :obj:`True`):
            이 프로세스가 main process인지 여부(logging/save를 수행해야 하는지 여부를 결정하는 데 사용됨)
    """
    assert len(predictions) == 2, '`predictions` should be a tuple with two elements (start_logits, end_logits).'
    all_start_logits, all_end_logits = predictions

    assert len(predictions[0]) == len(features), f'Got {len(predictions[0])} predictions and {len(features)} features.'

    # example과 mapping되는 feature 생성
    print('postprocess_qa ' + str(examples))
    example_id_to_index = {k: i for i, k in enumerate(examples[ID])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature[EXAMPLE_ID]]].append(i)

    # prediction, nbest에 해당하는 OrderedDict 생성합니다.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()

    # Logging.
    logger = setup_logger('postprocess log')

    # 전체 example들에 대한 main Loop
    for example_index, example in enumerate(tqdm(examples)):
        # 해당하는 현재 example index
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        # 현재 example에 대한 모든 feature 생성합니다.
        for feature_index in feature_indices:
            # 각 feature에 대한 모든 prediction을 가져옵니다.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # logit과 original context의 logit을 mapping합니다.
            offset_mapping = features[feature_index][OFFSET_MAPPING]
            # Optional : `token_is_max_context`, 제공되는 경우 현재 기능에서 사용할 수 있는 max context가 없는 answer를 제거합니다
            token_is_max_context = features[feature_index].get('token_is_max_context', None)

            # minimum null prediction을 업데이트 합니다.
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction[SCORE] > feature_null_score:
                min_null_prediction = {
                    OFFSET_MAPPING: (0, 0),
                    SCORE: feature_null_score,
                    START_LOGIT: start_logits[0],
                    END_LOGIT: end_logits[0],
                }

            # `n_best_size`보다 큰 start and end logits을 살펴봅니다.
            start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # out-of-scope answers는 고려하지 않습니다.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # 길이가 < 0 또는 > max_answer_length인 answer도 고려하지 않습니다.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # 최대 context가 없는 answer도 고려하지 않습니다.
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue
                    prelim_predictions.append(
                        {
                            OFFSET_MAPPING: (
                                offset_mapping[start_index][0],
                                offset_mapping[end_index][1],
                            ),
                            SCORE: start_logits[start_index] + end_logits[end_index],
                            START_LOGIT: start_logits[start_index],
                            END_LOGIT: end_logits[end_index],
                        },
                    )

        if version_2_with_negative:
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction[SCORE]

        # 가장 좋은 `n_best_size` predictions만 유지합니다.
        predictions_result = sorted(prelim_predictions, key=lambda x: x[SCORE], reverse=True)[:n_best_size]

        # 낮은 점수로 인해 제거된 경우 minimum null prediction을 다시 추가합니다.
        if version_2_with_negative and not any(p[OFFSET_MAPPING] == (0, 0) for p in predictions_result):
            predictions_result.append(min_null_prediction)

        # offset을 사용하여 original context에서 answer text를 수집합니다.
        context = example[CONTEXT]
        for pred in predictions_result:
            offsets = pred.pop(OFFSET_MAPPING)
            pred[TEXT] = context[offsets[0]: offsets[1]]

        # rare edge case에는 null이 아닌 예측이 하나도 없으며 failure를 피하기 위해 fake prediction을 만듭니다.
        if len(predictions_result) == 0 or (len(predictions_result) == 1 and predictions_result[0][TEXT] == ''):
            predictions_result.insert(
                0, {TEXT: 'empty', START_LOGIT: 0.0, END_LOGIT: 0.0, SCORE: 0.0},
            )

        # 모든 점수의 소프트맥스를 계산합니다.
        scores = np.array([pred.pop(SCORE) for pred in predictions_result])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # 예측값에 확률을 포함합니다.
        for prob, pred in zip(probs, predictions_result):
            pred[PROBABILITY] = prob

        # best prediction을 선택합니다.
        if not version_2_with_negative:
            all_predictions[example[ID]] = predictions_result[0][TEXT]
        else:
            i = 0
            while predictions_result[i][TEXT] == '':
                i += 1
            best_non_null_pred = predictions_result[i]

            score_diff = null_score - best_non_null_pred[START_LOGIT] - best_non_null_pred[END_LOGIT]
            scores_diff_json[example[ID]] = float(score_diff)
            if score_diff > null_score_diff_threshold:
                all_predictions[example[ID]] = ''
            else:
                all_predictions[example[ID]] = best_non_null_pred[TEXT]

        all_nbest_json[example[ID]] = [
            {
                k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v)
                for k, v in pred.items()
            }
            for pred in predictions_result
        ]

    # output_dir이 있으면 모든 dicts를 저장합니다.
    if output_dir is not None:
        assert os.path.isdir(output_dir), f'{output_dir} is not a directory.'

        prediction_file = os.path.join(output_dir, f'{PREDICTIONS}.json' if prefix is None else f'{PREDICTIONS}_{prefix}.json')
        nbest_file = os.path.join(output_dir, f'{NBEST_PREDICTIONS}.json' if prefix is None else f'{NBEST_PREDICTIONS}_{prefix}.json')
        if version_2_with_negative:
            null_odds_file = os.path.join(output_dir, f'{NULL_SCORE_DIFF}.json' if prefix is None else f'{NULL_SCORE_DIFF}_{prefix}.json')

        logger.info(f'Saving predictions to {prediction_file}.')
        with open(prediction_file, 'w', encoding='utf-8') as writer:
            writer.write(json.dumps(all_predictions, indent=4, ensure_ascii=False) + '\n')

        logger.info(f'Saving nbest_preds to {nbest_file}.')
        with open(nbest_file, 'w', encoding='utf-8') as writer:
            writer.write(json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + '\n')

        if version_2_with_negative:
            logger.info(f'Saving null_odds to {null_odds_file}.')
            with open(null_odds_file, 'w', encoding='utf-8') as writer:
                writer.write(json.dumps(scores_diff_json, indent=4, ensure_ascii=False) + '\n')

    return all_predictions
