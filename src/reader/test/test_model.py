# 모델이 올바르게 로드되는지 확인
# 잘못된 경로를 입력했을 때 예외 처리가 되는지 확인
def test_load_model():
    model_args = ModelArguments(model_name_or_path='bert-base-uncased')
    reader = Reader(model_args)
    reader.load_model()
    assert reader.model is not None


# 토크나이저가 정상적으로 로드되는지 확인
def test_load_tokenizer():
    model_args = ModelArguments(model_name_or_path='bert-base-uncased')
    reader = Reader(model_args)
    reader.load_tokenizer()
    assert reader.tokenizer is not None


# 모델의 forward 패스를 통해 올바른 출력을 생성하는지 확인
# 입력이 잘못되었을 때의 예외 처리도 확인
def test_forward():
    model_args = ModelArguments(model_name_or_path='bert-base-uncased')
    reader = Reader(model_args)
    reader.load_model()
    input_ids = torch.tensor([[101, 2003, 102]])
    attention_mask = torch.tensor([[1, 1, 1]])
    token_type_ids = torch.tensor([[0, 0, 0]])
    output = reader.forward(input_ids, attention_mask, token_type_ids)
    assert isinstance(output, dict)
