from abc import ABC
from abc import abstractmethod

from datasets import load_from_disk
from transformers import AutoTokenizer

from src.reader.utils.arguments import DataTrainingArguments


class DataHandler(ABC):
    def __init__(self, data_args: DataTrainingArguments, tokenizer: AutoTokenizer) -> None:
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.max_seq_length = min(
            data_args.max_seq_length, tokenizer.model_max_length,
        )

        self.datasets = load_from_disk(self.data_args.dataset_name)

    def process(self, type: str) -> dict:
        processed_data = self.handle_features(self.datasets, type)
        return processed_data

    def load_data(self, type: str) -> dict:
        if type not in ['train', 'validation']:
            raise ValueError("Invalid type: must be 'train' or 'validation'")
        datasets = self.process(type)

        return datasets

    @abstractmethod
    def handle_features(self, datasets: dict, type: str) -> dict:
        pass


class DataPreProcessor(DataHandler):
    def handle_features(self, datasets: dict, type: str) -> dict:
        if type == 'train':
            column_names = datasets['train'].column_names
        else:
            column_names = datasets['validation'].column_names

        self.question_column_name = 'question' if 'question' in column_names else column_names[0]
        self.context_column_name = 'context' if 'context' in column_names else column_names[1]
        self.answer_column_name = 'answers' if 'answers' in column_names else column_names[2]

        dataset = datasets[type].map(
            self.prepare_train_features if type == 'train' else self.prepare_validation_features,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not self.data_args.overwrite_cache,
        )

        return dataset

    def prepare_train_features(self, examples):
        tokenizer = self.tokenizer
        pad_on_right = tokenizer.padding_side == 'right'
        tokenized_examples = tokenizer(
            examples[self.question_column_name if pad_on_right else self.context_column_name],
            examples[self.context_column_name if pad_on_right else self.question_column_name],
            truncation='only_second' if pad_on_right else 'only_first',
            max_length=self.max_seq_length,
            stride=self.data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding='max_length' if self.data_args.pad_to_max_length else False,
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
            answers = examples[self.answer_column_name][sample_index]

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

    def prepare_validation_features(self, examples):
        tokenizer = self.tokenizer
        pad_on_right = tokenizer.padding_side == 'right'
        tokenized_examples = tokenizer(
            examples[self.question_column_name if pad_on_right else self.context_column_name],
            examples[self.context_column_name if pad_on_right else self.question_column_name],
            truncation='only_second' if pad_on_right else 'only_first',
            max_length=self.max_seq_length,
            stride=self.data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding='max_length' if self.data_args.pad_to_max_length else False,
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


class DataPostProcessor(DataHandler):
    def handle_features(self, datasets: dict, type: str) -> dict:
        return {'data': 'post-processed'}
