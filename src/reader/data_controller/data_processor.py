from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from datasets import DatasetDict
from transformers import AutoTokenizer
from transformers import EvalPrediction

from src.reader.data_controller.postprocess_qa import postprocess_qa_predictions
from src.utils.constants import key_names
from src.utils.constants import reader_configuration

class DataProcessor(ABC):
    name = key_names.DATA_PROCESSOR

    @classmethod
    @abstractmethod
    def process(cls, data_args, *datasets: DatasetDict, tokenizer: AutoTokenizer | None = None):
        pass


class DataPreProcessor(DataProcessor):
    name = key_names.PREPROCESSOR

    @classmethod
    def process(cls, data_args, *datasets: DatasetDict, tokenizer: AutoTokenizer | None = None) -> DatasetDict:
        """데이터 전처리

        Args:
            type : 처리할 데이터 타입/방법 등
            data_args : dataarguments
            *datasets : 데이터셋


        Returns:
            processed dataset
        """
        print('Pre-processing...')
        dataset = datasets[0]

        dataset_column_names = dataset.column_names

        def prepare_train_features(examples):
            question_column_name = key_names.QUESTION if key_names.QUESTION in dataset_column_names else dataset_column_names[0]
            context_column_name = key_names.CONTEXT if key_names.CONTEXT in dataset_column_names else dataset_column_names[1]
            answer_column_name = key_names.ANSWERS if key_names.ANSWERS in dataset_column_names else dataset_column_names[2]

            pad_on_right = tokenizer.padding_side == reader_configuration.TOKENIZER_PADDING_DIRECTION
            tokenized_examples = tokenizer(
                examples[question_column_name if pad_on_right else context_column_name],
                examples[context_column_name if pad_on_right else question_column_name],
                truncation=reader_configuration.TOKENIZER_TRUNCATE_SECOND if pad_on_right else reader_configuration.TOKENIZER_TRUNCATE_SECOND,
                max_length=min(
                    data_args.max_seq_length, tokenizer.model_max_length,
                ),
                stride=data_args.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding=reader_configuration.PADDING_OPTION if data_args.pad_to_max_length else False,
            )

            sample_mapping = tokenized_examples.pop(key_names.READER_SAMPLE_MAPPING)
            offset_mapping = tokenized_examples.pop(key_names.READER_OFFSET_MAPPING)

            tokenized_examples[key_names.MODEL_INFERED_START] = []
            tokenized_examples[key_names.MODEL_INFERED_END] = []

            for i, offsets in enumerate(offset_mapping):
                input_ids = tokenized_examples[key_names.TOKEN_INPUT_IDS][i]
                cls_index = input_ids.index(tokenizer.cls_token_id)

                sequence_ids = tokenized_examples.sequence_ids(i)
                sample_index = sample_mapping[i]
                answers = examples[answer_column_name][sample_index]

                if len(answers[key_names.ANSWER_START]) == 0:
                    tokenized_examples[key_names.MODEL_INFERED_START].append(cls_index)
                    tokenized_examples[key_names.MODEL_INFERED_END].append(cls_index)
                else:
                    start_char = answers[key_names.ANSWER_START][0]
                    end_char = start_char + len(answers[key_names.ANSWER_TEXT][0])

                    token_start_index = 0
                    while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                        token_start_index += 1

                    token_end_index = len(input_ids) - 1
                    while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                        token_end_index -= 1

                    if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                        tokenized_examples[key_names.MODEL_INFERED_START].append(cls_index)
                        tokenized_examples[key_names.MODEL_INFERED_END].append(cls_index)
                    else:
                        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                            token_start_index += 1
                        tokenized_examples[key_names.MODEL_INFERED_START].append(token_start_index - 1)

                        while offsets[token_end_index][1] >= end_char:
                            token_end_index -= 1
                        tokenized_examples[key_names.MODEL_INFERED_END].append(token_end_index + 1)

            return tokenized_examples

        def prepare_validation_features(examples):
            question_column_name = key_names.QUESTION if key_names.QUESTION in dataset_column_names else dataset_column_names[0]
            context_column_name = key_names.CONTEXT if key_names.CONTEXT in dataset_column_names else dataset_column_names[1]

            pad_on_right = tokenizer.padding_side == reader_configuration.TOKENIZER_PADDING_DIRECTION

            tokenized_examples = tokenizer(
                examples[question_column_name if pad_on_right else context_column_name],
                examples[context_column_name if pad_on_right else question_column_name],
                truncation=reader_configuration.TOKENIZER_TRUNCATE_SECOND if pad_on_right else reader_configuration.TOKENIZER_TRUNCATE_FIRST,
                max_length=data_args.max_seq_length,
                stride=data_args.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding=reader_configuration.TOKENIZER_PADDING_DIRECTION if data_args.pad_to_max_length else False,
            )

            sample_mapping = tokenized_examples.pop(key_names.READER_SAMPLE_MAPPING)
            tokenized_examples[key_names.EXAMPLE_ID] = []

            for i in range(len(tokenized_examples[key_names.TOKEN_INPUT_IDS])):
                sequence_ids = tokenized_examples.sequence_ids(i)
                context_index = 1 if pad_on_right else 0

                sample_index = sample_mapping[i]
                tokenized_examples[key_names.EXAMPLE_ID].append(examples[key_names.ID][sample_index])

                tokenized_examples[key_names.READER_OFFSET_MAPPING][i] = [
                    (o if sequence_ids[k] == context_index else None)
                    for k, o in enumerate(tokenized_examples[key_names.READER_OFFSET_MAPPING][i])
                ]
            return tokenized_examples

        dataset = dataset.map(
            prepare_train_features if data_args.do_train else prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=key_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        print('preprocessed '+str(dataset))
        return dataset


class DataPostProcessor(DataProcessor):
    name = key_names.POSTPROCESSOR

    @classmethod
    def process(cls, data_args, *datasets, tokenizer: AutoTokenizer | None = None):
        # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
        predictions = postprocess_qa_predictions(
            examples=datasets[0],
            features=datasets[1],
            predictions=datasets[2],
            max_answer_length=data_args.max_answer_length,
            output_dir=data_args.output_dir,
        )
        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
        formatted_predictions = [
            {key_names.ID: k, key_names.PREDICTION_TEXT: v} for k, v in predictions.items()
        ]
        if data_args.do_predict:
            return formatted_predictions

        elif data_args.do_eval:
            answer_column_name = (
                key_names.ANSWERS if key_names.ANSWERS in datasets[0].column_names else datasets[0].column_names[2]
            )
            references = [
                {key_names.ID: ex[key_names.ID], key_names.ANSWERS: ex[answer_column_name]}
                for ex in datasets[0]
            ]
            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references,
            )
