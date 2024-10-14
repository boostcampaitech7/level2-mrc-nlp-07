from arguments import DataTrainingArguments


# 반환된 데이터가 예상한 형식(dict)인지 검증
def test_load_data():
    data_args = DataTrainingArguments(dataset_name='test_dataset')
    data_handler = DataHandler(data_args)
    data = data_handler.load_data()
    assert isinstance(data, dict)
    assert 'data' in data


# 데이터 처리 결과가 예상한 형식(dict)으로 나오는지 검증
def test_process_data():
    data_args = DataTrainingArguments()
    data_handler = DataHandler(data_args)
    processed_data = data_handler.process_data()
    assert isinstance(processed_data, dict)
    assert 'data' in processed_data
