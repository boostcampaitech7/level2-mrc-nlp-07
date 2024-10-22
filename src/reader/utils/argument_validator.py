from __future__ import annotations


class InvalidArgumentError(Exception):
    pass


def validate_flags(do_train: bool, do_eval: bool, do_predict: bool):
    if (do_train and do_predict) or (do_predict and do_eval):
        raise InvalidArgumentError(
            '학습과 예측은 동시에 진행될 수 없습니다. --do_train/--do_eval과 --do_predict 중 하나를 빼고 실행해주세요.',
        )
