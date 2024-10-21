class InvalidArgumentError(Exception):
    pass


def validate_flags(do_train: bool, do_eval: bool, do_predict: bool):
    if (do_train and do_predict) or (do_predict and do_eval):
        raise InvalidArgumentError(
            '학습과 예측은 동시에 진행될 수 없습니다. --do_train/--do_eval과 --do_predict 중 하나를 빼고 실행해주세요.',
        )
    if do_eval and do_train:
        raise InvalidArgumentError(
            '--do_eval에는 학습 과정이 포함되어 있습니다. 학습만 원하시는 경우에는 --do_train 인자로 실행해주세요.',
        )
