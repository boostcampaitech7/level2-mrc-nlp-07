from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from datasets import DatasetDict
from transformers import AutoTokenizer
from transformers import EvalPrediction

from src.reader.data_controller.postprocess_qa import postprocess_qa_predictions
# TODO: DataProcessor에 BatchEncoding 형식 적용
# from transformers import BatchEncoding


class DataProcessor(ABC):
    name = 'DataProc'

    @classmethod
    @abstractmethod
    def process(cls, data_args, *datasets: DatasetDict, tokenizer: AutoTokenizer | None = None):
        pass


class DataPreProcessor(DataProcessor):
    name = 'pre'

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

        column_names = dataset.column_names

        def prepare_train_features(examples):
            question_column_name = 'question' if 'question' in column_names else column_names[0]
            context_column_name = 'context' if 'context' in column_names else column_names[1]
            answer_column_name = 'answers' if 'answers' in column_names else column_names[2]

            pad_on_right = tokenizer.padding_side == 'right'
            tokenized_examples = tokenizer(
                examples[question_column_name if pad_on_right else context_column_name],
                examples[context_column_name if pad_on_right else question_column_name],
                truncation='only_second' if pad_on_right else 'only_first',
                max_length=min(
                    data_args.max_seq_length, tokenizer.model_max_length,
                ),
                stride=data_args.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding='max_length' if data_args.pad_to_max_length else False,
            )

            sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')
            offset_mapping = tokenized_examples.pop('offset_mapping')

            tokenized_examples['start_positions'] = []
            tokenized_examples['end_positions'] = []

            for i, offsets in enumerate(offset_mapping):
                input_ids = tokenized_examples['input_ids'][i]
                cls_index = input_ids.index(tokenizer.cls_token_id)

                sequence_ids = tokenized_examples.sequence_ids(i)
                sample_index = sample_mapping[i]
                answers = examples[answer_column_name][sample_index]

                if len(answers['answer_start']) == 0:
                    tokenized_examples['start_positions'].append(cls_index)
                    tokenized_examples['end_positions'].append(cls_index)
                else:
                    start_char = answers['answer_start'][0]
                    end_char = start_char + len(answers['text'][0])

                    token_start_index = 0
                    while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                        token_start_index += 1

                    token_end_index = len(input_ids) - 1
                    while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                        token_end_index -= 1

                    if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                        tokenized_examples['start_positions'].append(cls_index)
                        tokenized_examples['end_positions'].append(cls_index)
                    else:
                        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                            token_start_index += 1
                        tokenized_examples['start_positions'].append(token_start_index - 1)

                        while offsets[token_end_index][1] >= end_char:
                            token_end_index -= 1
                        tokenized_examples['end_positions'].append(token_end_index + 1)

            return tokenized_examples

        def prepare_validation_features(examples):
            question_column_name = 'question' if 'question' in column_names else column_names[0]
            context_column_name = 'context' if 'context' in column_names else column_names[1]

            pad_on_right = tokenizer.padding_side == 'right'

            tokenized_examples = tokenizer(
                examples[question_column_name if pad_on_right else context_column_name],
                examples[context_column_name if pad_on_right else question_column_name],
                truncation='only_second' if pad_on_right else 'only_first',
                max_length=data_args.max_seq_length,
                stride=data_args.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding='max_length' if data_args.pad_to_max_length else False,
            )

            sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')
            tokenized_examples['example_id'] = []

            for i in range(len(tokenized_examples['input_ids'])):
                sequence_ids = tokenized_examples.sequence_ids(i)
                context_index = 1 if pad_on_right else 0

                sample_index = sample_mapping[i]
                tokenized_examples['example_id'].append(examples['id'][sample_index])

                tokenized_examples['offset_mapping'][i] = [
                    (o if sequence_ids[k] == context_index else None)
                    for k, o in enumerate(tokenized_examples['offset_mapping'][i])
                ]
            return tokenized_examples

        dataset = dataset.map(
            prepare_train_features if data_args.do_train else prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        return dataset


class DataPostProcessor(DataProcessor):
    name = 'post'

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
            {'id': k, 'prediction_text': v} for k, v in predictions.items()
        ]
        if data_args.do_predict:
            return formatted_predictions

        elif data_args.do_eval:
            answer_column_name = (
                'answers' if 'answers' in datasets[0].column_names else datasets[0].column_names[2]
            )
            references = [
                {'id': ex['id'], 'answers': ex[answer_column_name]}
                for ex in datasets[0]
                # LOOK 후처리로 넘어오는 데이터셋 3총사 구조 파악 필요
            ]
            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references,
            )


'''if __name__ == '__main__':
    data = BatchEncoding(
        {
            'input_ids': [101, 102, 103],
            'attention_mask': [1, 1, 1],
        },
    )

    preprocessed_data = DataPreProcessor.process(data)
    print(preprocessed_data)

    postprocessed_data = DataPostProcessor.process(data)
    print(postprocessed_data)
'''
