import collections
import json
import logging
import numpy as np
import os
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union

class PostProcessor:
    def __init__(self, 
                 version_2_with_negative: bool = False, 
                 n_best_size: int = 20, 
                 max_answer_length: int = 30, 
                 null_score_diff_threshold: float = 0.0,
                 output_dir: Optional[str] = None, 
                 prefix: Optional[str] = None, 
                 is_world_process_zero: bool = True):
        self.version_2_with_negative = version_2_with_negative
        self.n_best_size = n_best_size
        self.max_answer_length = max_answer_length
        self.null_score_diff_threshold = null_score_diff_threshold
        self.output_dir = output_dir
        self.prefix = prefix
        self.is_world_process_zero = is_world_process_zero
        self.logger = logging.getLogger(__name__)

    def post_process(self, examples: List[Dict], features: List[Dict], predictions: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Union[str, List[Dict]]]:
        all_start_logits, all_end_logits = self._validate_predictions(predictions, features)
        example_id_to_index, features_per_example = self._map_examples_to_features(examples, features)
        all_predictions, all_nbest_json = self._initialize_predictions()
        
        if self.version_2_with_negative:
            scores_diff_json = collections.OrderedDict()

        self.logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

        for example_index, example in enumerate(tqdm(examples)):
            feature_indices = features_per_example[example_index]
            min_null_prediction, prelim_predictions = self._get_preliminary_predictions(feature_indices, all_start_logits, all_end_logits, features, example)

            if self.version_2_with_negative:
                prelim_predictions.append(min_null_prediction)
                null_score = min_null_prediction["score"]

            predictions = self._select_best_predictions(prelim_predictions)
            self._add_context_to_predictions(predictions, example)

            if self.version_2_with_negative:
                best_non_null_pred, score_diff = self._evaluate_null_prediction(predictions, null_score)
                all_predictions[example["id"]] = best_non_null_pred["text"] if score_diff <= self.null_score_diff_threshold else ""
                scores_diff_json[example["id"]] = float(score_diff)
            else:
                all_predictions[example["id"]] = predictions[0]["text"]

            all_nbest_json[example["id"]] = self._format_nbest_predictions(predictions)

        if self.output_dir:
            self._save_results(all_predictions, all_nbest_json, scores_diff_json if self.version_2_with_negative else None)

        return all_predictions

    def _validate_predictions(self, predictions: Tuple[np.ndarray, np.ndarray], features: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        assert len(predictions) == 2, "`predictions` should be a tuple with two elements (start_logits, end_logits)."
        all_start_logits, all_end_logits = predictions
        assert len(all_start_logits) == len(features), f"Got {len(all_start_logits)} predictions and {len(features)} features."
        return all_start_logits, all_end_logits

    def _map_examples_to_features(self, examples: List[Dict], features: List[Dict]) -> Tuple[Dict[str, int], Dict[int, List[int]]]:
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)
        return example_id_to_index, features_per_example

    def _initialize_predictions(self) -> Tuple[Dict[str, str], Dict[str, List[Dict]]]:
        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
        return all_predictions, all_nbest_json

    def _get_preliminary_predictions(self, feature_indices: List[int], all_start_logits: np.ndarray, all_end_logits: np.ndarray, features: List[Dict], example: Dict) -> Tuple[Optional[Dict[str, Union[Tuple[int, int], float]]], List[Dict[str, Union[Tuple[int, int], float]]]]:
        min_null_prediction = None
        prelim_predictions = []

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            token_is_max_context = features[feature_index].get("token_is_max_context", None)

            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            start_indexes = np.argsort(start_logits)[-1 : -self.n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -self.n_best_size - 1 : -1].tolist()

            self._collect_prelim_predictions(prelim_predictions, start_indexes, end_indexes, offset_mapping, token_is_max_context, start_logits, end_logits)

        return min_null_prediction, prelim_predictions

    def _collect_prelim_predictions(self, prelim_predictions: List[Dict[str, Union[Tuple[int, int], float]]], start_indexes: List[int], end_indexes: List[int], offset_mapping: List[Tuple[int, int]], token_is_max_context: Optional[Dict[str, bool]], start_logits: np.ndarray, end_logits: np.ndarray) -> None:
        for start_index in start_indexes:
            for end_index in end_indexes:
                if self._is_valid_index(start_index, end_index, offset_mapping):
                    prelim_predictions.append({
                        "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                        "score": start_logits[start_index] + end_logits[end_index],
                        "start_logit": start_logits[start_index],
                        "end_logit": end_logits[end_index],
                    })

    def _is_valid_index(self, start_index: int, end_index: int, offset_mapping: List[Tuple[int, int]]) -> bool:
        return (start_index < len(offset_mapping) and 
                end_index < len(offset_mapping) and 
                offset_mapping[start_index] is not None and 
                offset_mapping[end_index] is not None and 
                end_index >= start_index and 
                end_index - start_index + 1 <= self.max_answer_length)

    def _select_best_predictions(self, prelim_predictions: List[Dict[str, Union[Tuple[int, int], float]]]) -> List[Dict[str, Union[Tuple[int, int], float]]]:
        return sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:self.n_best_size]

    def _add_context_to_predictions(self, predictions: List[Dict[str, Union[Tuple[int, int], float]]], example: Dict) -> None:
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0]: offsets[1]]

    def _evaluate_null_prediction(self, predictions: List[Dict[str, Union[Tuple[int, int], float]]], null_score: float) -> Tuple[Dict[str, Union[Tuple[int, int], float]], float]:
        best_non_null_pred = next(pred for pred in predictions if pred["text"] != "")
        score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
        return best_non_null_pred, score_diff

    def _format_nbest_predictions(self, predictions: List[Dict[str, Union[Tuple[int, int], float]]]) -> List[Dict]:
        return [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in predictions
        ]

    def _save_results(self, all_predictions: Dict[str, str], all_nbest_json: Dict[str, List[Dict]], scores_diff_json: Optional[Dict[str, float]] = None) -> None:
        prediction_file = os.path.join(self.output_dir, f"predictions_{self.prefix}.json" if self.prefix else "predictions.json")
        nbest_file = os.path.join(self.output_dir, f"nbest_predictions_{self.prefix}.json" if self.prefix else "nbest_predictions.json")
        if self.version_2_with_negative:
            null_odds_file = os.path.join(self.output_dir, f"null_odds_{self.prefix}.json" if self.prefix else "null_odds.json")

        self.logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n")
        
        self.logger.info(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + "\n")

        if self.version_2_with_negative:
            self.logger.info(f"Saving null_odds to {null_odds_file}.")
            with open(null_odds_file, "w", encoding="utf-8") as writer:
                writer.write(json.dumps(scores_diff_json, indent=4, ensure_ascii=False) + "\n")
