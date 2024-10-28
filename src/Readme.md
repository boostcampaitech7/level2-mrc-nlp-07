reader() : retriever() 코드와 연동하기 위해, csv를 로드해 전처리후 모델을 작동시킵니다.

SAVE_TRAIN_PATH로 저장된 위치의 csv를 datasetdict로 읽고, 필요 없는 칼럼 삭제, answers 필드 파싱등을 한 후 태스크를 시작합니다.

training_args.do_train, training_args.do_eval, training_args.do_predict 셋중 하나를 True로 바꿈으로 학습/평가/추론이 가능하고 model_args.model_name_or_path로 학습/추론에 사용할 모델을 지정 가능합니다.
