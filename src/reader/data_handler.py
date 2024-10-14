from __future__ import annotations

from arguments import DataTrainingArguments


class DataHandler:
    def __init__(self, data_args: DataTrainingArguments) -> None:
        pass

    def load_data(self) -> dict:
        # TODO: 데이터 로드 로직
        return {'data': 'loaded'}

    def process_data(self) -> dict:
        # TODO: 데이터 처리 로직
        return {'data': 'processed'}
