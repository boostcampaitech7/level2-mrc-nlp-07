def check_original_in_context(df):
    return df["original_context"] in df["retrieval_context"]

def calculate_reverse_rank_score(df):
    try:
        rank = df["retrieval_context"].index(df["original_context"]) + 1  # 1등부터 시작하는 등수로 변환
        score = 1 / rank  # RR score
    except ValueError: # 정답이 후보에 없을 경우
        score = 0.0
    return score    

def calculate_linear_score(df):
    candidate = len(df["retrieval_context"])
    if check_original_in_context(df):
        rank = df["retrieval_context"].index(df["original_context"]) 
        score = (candidate - rank) / candidate  # linear score
    else: # 정답이 후보에 없을 경우
        score = 0 
    return score

