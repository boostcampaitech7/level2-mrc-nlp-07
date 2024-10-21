from helper import TrainingValidator
from typing import Tuple, Any, Dict

class DataPreprocessor:
    def __init__(self, tokenizer: Any, data_args: Any, training_args: Any, datasets: Dict[str, Any] = None):
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.training_args = training_args
        # datasets를 인자로 받아 TrainingValidator 초기화
        self.validator = TrainingValidator(data_args, training_args, datasets, tokenizer)

    def prepare_datasets(self, datasets: Dict[str, Any]) -> Tuple[Any, Any, Any]:
        # datasets 할당
        self.validator.datasets = datasets
        if self.training_args.do_train:
            column_names = datasets["train"].column_names
        else:
            column_names = datasets["validation"].column_names
        question_column_name = "question" if "question" in column_names else column_names[0]
        context_column_name = "context" if "context" in column_names else column_names[1]
        answer_column_name = "answers" if "answers" in column_names else column_names[2]

        pad_on_right = self.tokenizer.padding_side == "right"

        # Check for any errors
        last_checkpoint, max_seq_length = self.validator.check_no_error()

        train_dataset = self.prepare_dataset(datasets, "train", question_column_name, context_column_name, answer_column_name, pad_on_right, max_seq_length) if self.training_args.do_train else None
        eval_dataset = self.prepare_dataset(datasets, "validation", question_column_name, context_column_name, answer_column_name, pad_on_right, max_seq_length) if self.training_args.do_eval else None

        return train_dataset, eval_dataset, last_checkpoint

    def prepare_dataset(self, datasets: Dict[str, Any], mode: str, question_column_name: str, context_column_name: str, answer_column_name: str, pad_on_right: bool, max_seq_length: int) -> Any:
        return datasets[mode].map(
            lambda examples: self.prepare_features(examples, question_column_name, context_column_name, answer_column_name, pad_on_right, max_seq_length),
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=datasets[mode].column_names,
            load_from_cache_file=not self.data_args.overwrite_cache,
        )

    def prepare_features(self, examples: Dict[str, Any], question_column_name: str, context_column_name: str, answer_column_name: str, pad_on_right: bool, max_seq_length: int) -> Dict[str, Any]:
        # Tokenization 과정
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

    def prepare_train_features(self, tokenized_examples: Dict[str, Any], examples: Dict[str, Any], sample_mapping: Any, answer_column_name: str, pad_on_right: bool) -> None:
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(tokenized_examples["offset_mapping"]):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]

            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                token_start_index, token_end_index = self.get_token_indices(offsets, sequence_ids, pad_on_right, input_ids)

                # Token start/end 인덱스가 유효한지 확인
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while token_end_index >= 0 and offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

    def prepare_eval_features(self, tokenized_examples: Dict[str, Any], examples: Dict[str, Any], sample_mapping: Any, pad_on_right: bool) -> None:
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

    def get_token_indices(self, offsets: Any, sequence_ids: Any, pad_on_right: bool, input_ids: Any) -> Tuple[int, int]:
        token_start_index = 0
        while token_start_index < len(sequence_ids) and sequence_ids[token_start_index] != (1 if pad_on_right else 0):
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while token_end_index >= 0 and sequence_ids[token_end_index] != (1 if pad_on_right else 0):
            token_end_index -= 1
        
        return token_start_index, token_end_index
