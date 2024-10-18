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
        self.version_2_with_negative: bool = version_2_with_negative
        self.n_best_size: int = n_best_size
        self.max_answer_length: int = max_answer_length
        self.null_score_diff_threshold: float = null_score_diff_threshold
        self.output_dir: Optional[str] = output_dir
        self.prefix: Optional[str] = prefix
        self.is_world_process_zero: bool = is_world_process_zero
        self.logger: logging.Logger = logging.getLogger(__name__)

    def post_process(self, examples: List[Dict], features: List[Dict], predictions: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Union[str, List[Dict]]]:
        assert len(predictions) == 2, "`predictions` should be a tuple with two elements (start_logits, end_logits)."
        all_start_logits, all_end_logits = predictions

        assert len(predictions[0]) == len(features), f"Got {len(predictions[0])} predictions and {len(features)} features."

        # Example과 mapping되는 feature 생성
        example_id_to_index: Dict[str, int] = {k: i for i, k in enumerate(examples["id"])}
        features_per_example: Dict[int, List[int]] = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        # Prediction 및 nbest에 해당하는 OrderedDict 생성
        all_predictions: Dict[str, str] = collections.OrderedDict()
        all_nbest_json: Dict[str, List[Dict]] = collections.OrderedDict()
        if self.version_2_with_negative:
            scores_diff_json: Dict[str, float] = collections.OrderedDict()

        self.logger.setLevel(logging.INFO if self.is_world_process_zero else logging.WARN)
        self.logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

        for example_index, example in enumerate(tqdm(examples)):
            feature_indices = features_per_example[example_index]
            min_null_prediction: Optional[Dict[str, Union[Tuple[int, int], float]]] = None
            prelim_predictions: List[Dict[str, Union[Tuple[int, int], float]]] = []

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

                for start_index in start_indexes:
                    for end_index in end_indexes:
                        if start_index >= len(offset_mapping) or end_index >= len(offset_mapping) or offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                            continue
                        if end_index < start_index or end_index - start_index + 1 > self.max_answer_length:
                            continue
                        if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                            continue
                        prelim_predictions.append({
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        })

            if self.version_2_with_negative:
                prelim_predictions.append(min_null_prediction)
                null_score = min_null_prediction["score"]

            predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:self.n_best_size]
            if self.version_2_with_negative and not any(p["offsets"] == (0, 0) for p in predictions):
                predictions.append(min_null_prediction)

            context = example["context"]
            for pred in predictions:
                offsets = pred.pop("offsets")
                pred["text"] = context[offsets[0]: offsets[1]]

            if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
                predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

            scores = np.array([pred.pop("score") for pred in predictions])
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()

            for prob, pred in zip(probs, predictions):
                pred["probability"] = prob

            if not self.version_2_with_negative:
                all_predictions[example["id"]] = predictions[0]["text"]
            else:
                i = 0
                while predictions[i]["text"] == "":
                    i += 1
                best_non_null_pred = predictions[i]
                score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
                scores_diff_json[example["id"]] = float(score_diff)
                if score_diff > self.null_score_diff_threshold:
                    all_predictions[example["id"]] = ""
                else:
                    all_predictions[example["id"]] = best_non_null_pred["text"]

            all_nbest_json[example["id"]] = [
                {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
                for pred in predictions
            ]

        if self.output_dir is not None:
            assert os.path.isdir(self.output_dir), f"{self.output_dir} is not a directory."
            self._save_results(all_predictions, all_nbest_json, scores_diff_json if self.version_2_with_negative else None)

        return all_predictions

    def _save_results(self, all_predictions: Dict[str, str], all_nbest_json: Dict[str, List[Dict]], scores_diff_json: Optional[Dict[str, float]] = None) -> None:
        prediction_file = os.path.join(self.output_dir, "predictions.json" if self.prefix is None else f"predictions_{self.prefix}.json")
        nbest_file = os.path.join(self.output_dir, "nbest_predictions.json" if self.prefix is None else f"nbest_predictions_{self.prefix}.json")
        if self.version_2_with_negative:
            null_odds_file = os.path.join(self.output_dir, "null_odds.json" if self.prefix is None else f"null_odds_{self.prefix}.json")

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
