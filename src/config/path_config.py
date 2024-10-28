from pathlib import Path 


BASE = Path(__file__).resolve().parent.parent.parent
data_path = "./data/"
wiki_path = "filtered_wiki.json"
train_data = "./data/train_dataset"
test_data = "./data/test_dataset"
train_name = "train.csv"
test_name = "test.csv"
outputs = "./outputs"

DATA_PATH = Path(BASE, data_path)
WIKI_PATH = Path(BASE, wiki_path)
TRAIN_DATA_PATH = Path(BASE, train_data)
TEST_DATA_PATH = Path(BASE, test_data)
SAVE_TRAIN_PATH = Path(BASE, train_name)
SAVE_TEST_PATH = Path(BASE, test_name)