import logging
import os
import sys
import random
import numpy as np
import torch
import collections
import json
from typing import Tuple, Optional
from tqdm import tqdm
from arguments import DataTrainingArguments, ModelArguments
from datasets import load_from_disk, load_metric
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    TrainingArguments,
    HfArgumentParser,
)
from utils_qa import check_no_error


class MRCTrainer:
    def __init__(self, model_args: ModelArguments, data_args: DataTrainingArguments, training_args: TrainingArguments):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.logger = logging.getLogger(__name__)

        self.set_random_seed()
        self.datasets = load_from_disk(data_args.dataset_name)
        self.config = AutoConfig.from_pretrained(
            model_args.config_name or model_args.model_name_or_path
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name or model_args.model_name_or_path,
            use_fast=True,
        )
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=self.config,
        )
        self.data_collator = DataCollatorWithPadding(
            self.tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
        )

    def set_random_seed(self):
        seed = 2024
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def prepare_datasets(self):
        column_names = self.datasets["train"].column_names if self.training_args.do_train else self.datasets["validation"].column_names
        question_column_name = "question" if "question" in column_names else column_names[0]
        context_column_name = "context" if "context" in column_names else column_names[1]
        answer_column_name = "answers" if "answers" in column_names else column_names[2]

        # Padding 설정
        pad_on_right = self.tokenizer.padding_side == "right"
        
        last_checkpoint, max_seq_length = check_no_error(
            self.data_args, self.training_args, self.datasets, self.tokenizer
        )

        train_dataset = self.prepare_dataset("train", question_column_name, context_column_name, answer_column_name, pad_on_right, max_seq_length) if self.training_args.do_train else None
        eval_dataset = self.prepare_dataset("validation", question_column_name, context_column_name, answer_column_name, pad_on_right, max_seq_length) if self.training_args.do_eval else None

        return train_dataset, eval_dataset, last_checkpoint

    def prepare_dataset(self, mode: str, question_column_name: str, context_column_name: str, answer_column_name: str, pad_on_right: bool, max_seq_length: int):
        """훈련 또는 평가 데이터셋을 준비합니다."""
        return self.datasets[mode].map(
            lambda examples: self.prepare_features(examples, question_column_name, context_column_name, answer_column_name, pad_on_right, max_seq_length),
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=self.datasets[mode].column_names,
            load_from_cache_file=not self.data_args.overwrite_cache,
        )

    def prepare_features(self, examples, question_column_name: str, context_column_name: str, answer_column_name: str, pad_on_right: bool, max_seq_length: int):
        """
        공통된 특징 준비 메서드.
        훈련 또는 평가에 사용할 특징을 준비합니다.

        Args:
            examples: 원본 예제 데이터.
            question_column_name: 질문 컬럼 이름.
            context_column_name: 문맥 컬럼 이름.
            answer_column_name: 답변 컬럼 이름.
            pad_on_right: 패딩 방향.
            max_seq_length: 최대 시퀀스 길이.
        
        Returns:
            tokenized_examples: 전처리된 특징 데이터.
        """
        tokenized_examples = self.tokenizer(
            examples[question_column_name],
            examples[context_column_name],
            truncation="only_second",
            max_length=max_seq_length,
            stride=self.data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if self.data_args.pad_to_max_length else False,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        if self.training_args.do_train:
            self.prepare_train_features(tokenized_examples, examples, sample_mapping, answer_column_name, pad_on_right)
        else:
            self.prepare_eval_features(tokenized_examples, examples, sample_mapping, pad_on_right)

        return tokenized_examples

    def prepare_train_features(self, tokenized_examples, examples, sample_mapping, answer_column_name: str, pad_on_right: bool):
        """훈련 데이터를 위한 특징을 준비합니다."""
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(tokenized_examples["offset_mapping"]):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)  # cls index

            # sequence id를 설정합니다.
            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]

            # answer가 없을 경우 cls_index를 answer로 설정합니다.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # text에서 current span의 Start token index
                token_start_index, token_end_index = self.get_token_indices(offsets, sequence_ids, pad_on_right, input_ids)

                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

    def prepare_eval_features(self, tokenized_examples, examples, sample_mapping, pad_on_right: bool):
        """평가 데이터를 위한 특징을 준비합니다."""
        tokenized_examples["example_id"] = []
        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0
            sample_index = sample_mapping[i]
            
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

    def get_token_indices(self, offsets, sequence_ids, pad_on_right, input_ids):
        """토큰의 시작 및 종료 인덱스를 반환합니다."""
        token_start_index = 0
        while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
            token_end_index -= 1
        
        return token_start_index, token_end_index



    def train(self):
        train_dataset, eval_dataset, last_checkpoint = self.prepare_datasets()

        # Trainer 초기화
        trainer = QuestionAnsweringTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset if self.training_args.do_train else None,
            eval_dataset=eval_dataset if self.training_args.do_eval else None,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            post_process_function=self.post_processing_function,
            compute_metrics=self.compute_metrics,
        )

        if self.training_args.do_train:
            checkpoint = last_checkpoint or self.model_args.model_name_or_path
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()
            self.log_results(train_result, "train")

        if self.training_args.do_eval:
            metrics = trainer.evaluate()
            self.log_results(metrics, "eval")

    def post_processing_function(
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        version_2_with_negative: bool = False,
        n_best_size: int = 20,
        max_answer_length: int = 30,
        null_score_diff_threshold: float = 0.0,
        output_dir: Optional[str] = None,
        prefix: Optional[str] = None,
        is_world_process_zero: bool = True,
    ):
        """
        Post-processes : qa model의 prediction 값을 후처리하는 함수
        모델은 start logit과 end logit을 반환하기 때문에, 이를 기반으로 original text로 변경하는 후처리가 필요함

        Args:
            examples: 전처리 되지 않은 데이터셋 (see the main script for more information).
            features: 전처리가 진행된 데이터셋 (see the main script for more information).
            predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
                모델의 예측값 :start logits과 the end logits을 나타내는 two arrays              첫번째 차원은 :obj:`features`의 element와 갯수가 맞아야함.
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
                dictionary : predictions, n_best predictions (with their scores and logits) if:obj:`version_2_with_negative=True`,
                dictionary : the scores differences between best and null answers
            prefix (:obj:`str`, `optional`):
                dictionary에 `prefix`가 포함되어 저장됨
            is_world_process_zero (:obj:`bool`, `optional`, defaults to :obj:`True`):
                이 프로세스가 main process인지 여부(logging/save를 수행해야 하는지 여부를 결정하는 데 사용됨)
        """
        logger = logging.getLogger(__name__)

        assert (
            len(predictions) == 2
        ), "`predictions` should be a tuple with two elements (start_logits, end_logits)."
        all_start_logits, all_end_logits = predictions

        assert len(predictions[0]) == len(
            features
        ), f"Got {len(predictions[0])} predictions and {len(features)} features."

        # example과 mapping되는 feature 생성
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        # prediction, nbest에 해당하는 OrderedDict 생성합니다.
        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
        if version_2_with_negative:
            scores_diff_json = collections.OrderedDict()

        # Logging.
        logger.setLevel(logging.INFO if is_world_process_zero else logging.WARN)
        logger.info(
            f"Post-processing {len(examples)} example predictions split into {len(features)} features."
        )

        # 전체 example들에 대한 main Loop
        for example_index, example in enumerate(tqdm(examples)):
            # 해당하는 현재 example index
            feature_indices = features_per_example[example_index]

            min_null_prediction = None
            prelim_predictions = []

            # 현재 example에 대한 모든 feature 생성합니다.
            for feature_index in feature_indices:
                # 각 featureure에 대한 모든 prediction을 가져옵니다.
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]
                # logit과 original context의 logit을 mapping합니다.
                offset_mapping = features[feature_index]["offset_mapping"]
                # Optional : `token_is_max_context`, 제공되는 경우 현재 기능에서 사용할 수 있는 max context가 없는 answer를 제거합니다
                token_is_max_context = features[feature_index].get(
                    "token_is_max_context", None
                )

                # minimum null prediction을 업데이트 합니다.
                feature_null_score = start_logits[0] + end_logits[0]
                if (
                    min_null_prediction is None
                    or min_null_prediction["score"] > feature_null_score
                ):
                    min_null_prediction = {
                        "offsets": (0, 0),
                        "score": feature_null_score,
                        "start_logit": start_logits[0],
                        "end_logit": end_logits[0],
                    }

                # `n_best_size`보다 큰 start and end logits을 살펴봅니다.
                start_indexes = np.argsort(start_logits)[
                    -1 : -n_best_size - 1 : -1
                ].tolist()

                end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

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
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                        ):
                            continue
                        # 최대 context가 없는 answer도 고려하지 않습니다.
                        if (
                            token_is_max_context is not None
                            and not token_is_max_context.get(str(start_index), False)
                        ):
                            continue
                        prelim_predictions.append(
                            {
                                "offsets": (
                                    offset_mapping[start_index][0],
                                    offset_mapping[end_index][1],
                                ),
                                "score": start_logits[start_index] + end_logits[end_index],
                                "start_logit": start_logits[start_index],
                                "end_logit": end_logits[end_index],
                            }
                        )

            if version_2_with_negative:
                # minimum null prediction을 추가합니다.
                prelim_predictions.append(min_null_prediction)
                null_score = min_null_prediction["score"]

            # 가장 좋은 `n_best_size` predictions만 유지합니다.
            predictions = sorted(
                prelim_predictions, key=lambda x: x["score"], reverse=True
            )[:n_best_size]

            # 낮은 점수로 인해 제거된 경우 minimum null prediction을 다시 추가합니다.
            if version_2_with_negative and not any(
                p["offsets"] == (0, 0) for p in predictions
            ):
                predictions.append(min_null_prediction)

            # offset을 사용하여 original context에서 answer text를 수집합니다.
            context = example["context"]
            for pred in predictions:
                offsets = pred.pop("offsets")
                pred["text"] = context[offsets[0] : offsets[1]]

            # rare edge case에는 null이 아닌 예측이 하나도 없으며 failure를 피하기 위해 fake prediction을 만듭니다.
            if len(predictions) == 0 or (
                len(predictions) == 1 and predictions[0]["text"] == ""
            ):

                predictions.insert(
                    0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0}
                )

            # 모든 점수의 소프트맥스를 계산합니다(we do it with numpy to stay independent from torch/tf in this file, using the LogSumExp trick).
            scores = np.array([pred.pop("score") for pred in predictions])
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()

            # 예측값에 확률을 포함합니다.
            for prob, pred in zip(probs, predictions):
                pred["probability"] = prob

            # best prediction을 선택합니다.
            if not version_2_with_negative:
                all_predictions[example["id"]] = predictions[0]["text"]
            else:
                # else case : 먼저 비어 있지 않은 최상의 예측을 찾아야 합니다
                i = 0
                while predictions[i]["text"] == "":
                    i += 1
                best_non_null_pred = predictions[i]

                # threshold를 사용해서 null prediction을 비교합니다.
                score_diff = (
                    null_score
                    - best_non_null_pred["start_logit"]
                    - best_non_null_pred["end_logit"]
                )
                scores_diff_json[example["id"]] = float(score_diff)  # JSON-serializable 가능
                if score_diff > null_score_diff_threshold:
                    all_predictions[example["id"]] = ""
                else:
                    all_predictions[example["id"]] = best_non_null_pred["text"]

            # np.float를 다시 float로 casting -> `predictions`은 JSON-serializable 가능
            all_nbest_json[example["id"]] = [
                {
                    k: (
                        float(v)
                        if isinstance(v, (np.float16, np.float32, np.float64))
                        else v
                    )
                    for k, v in pred.items()
                }
                for pred in predictions
            ]

        # output_dir이 있으면 모든 dicts를 저장합니다.
        if output_dir is not None:
            assert os.path.isdir(output_dir), f"{output_dir} is not a directory."

            prediction_file = os.path.join(
                output_dir,
                "predictions.json" if prefix is None else f"predictions_{prefix}".json,
            )
            nbest_file = os.path.join(
                output_dir,
                "nbest_predictions.json"
                if prefix is None
                else f"nbest_predictions_{prefix}".json,
            )
            if version_2_with_negative:
                null_odds_file = os.path.join(
                    output_dir,
                    "null_odds.json" if prefix is None else f"null_odds_{prefix}".json,
                )

            logger.info(f"Saving predictions to {prediction_file}.")
            with open(prediction_file, "w", encoding="utf-8") as writer:
                writer.write(
                    json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n"
                )
            logger.info(f"Saving nbest_preds to {nbest_file}.")
            with open(nbest_file, "w", encoding="utf-8") as writer:
                writer.write(
                    json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + "\n"
                )
            if version_2_with_negative:
                logger.info(f"Saving null_odds to {null_odds_file}.")
                with open(null_odds_file, "w", encoding="utf-8") as writer:
                    writer.write(
                        json.dumps(scores_diff_json, indent=4, ensure_ascii=False) + "\n"
                    )

        return all_predictions


    def compute_metrics(self, p: EvalPrediction):
        metric = load_metric("squad")
        metrics = metric.compute(predictions=p.predictions, references=p.label_ids)
        return metrics
    
    def log_results(self, metrics, mode: str):
        output_file = os.path.join(self.training_args.output_dir, f"{mode}_results.txt")
        with open(output_file, "w") as writer:
            self.logger.info(f"***** {mode.capitalize()} results *****")
            for key, value in sorted(metrics.items()):
                self.logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")


def main():
    # Arguments 파싱
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    mrc_trainer = MRCTrainer(model_args, data_args, training_args)
    mrc_trainer.train()


if __name__ == "__main__":
    main()
