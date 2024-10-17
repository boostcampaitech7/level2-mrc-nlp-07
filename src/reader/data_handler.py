from datasets import load_from_disk
from transformers import AutoTokenizer

from src.reader.utils.arguments import DataTrainingArguments
from data_processor import DataProcessor

class DataHandler():
    def __init__(self, data_args: DataTrainingArguments, tokenizer: AutoTokenizer, postprocessor:DataProcessor, preprocessor:DataProcessor) -> None:
        """DataHandler 초기화 설정.
        Args:
            data_args (DataTrainingArguments): DataTrainingArguments 형식 
            tokenizer (AutoTokenizer): 토크나이저 설정
        """
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.max_seq_length = min(
            data_args.max_seq_length, tokenizer.model_max_length,
        )

        self.datasets = load_from_disk(self.data_args.dataset_name)
                
        self.processors = {'pre':preprocessor, 'pos':postprocessor(data_args, self.datasets)} 
        #TODO: 입력을 여러개 받고 해당 클래스의 정보를 읽어서 dictionary 등록하는 방식으로 변경


    def process_func(self, type: str) -> dict:
        """데이터를 처리하는 함수를 반환

        Args:
            type (str): 처리할 dataprocessor 정보

        Returns:
            processed
        """
        processor_func = self.processors[type].process
        return processor_func

    def process_data(self, type: str) -> dict:
        """데이터를 처리함

        Args:
            type (str): _description_

        Returns:
            BatchEncoding: _description_
        """
        processed_data = self.processors[type].process()
        return processed_data
    
    def load_data(self, type:str) -> dict:
        """데이터를 로드, 기본적으로 전처리 함

        Args:
            type (str): 로드할 데이터 종류를 지정, 'train'/'validation'

        Returns:
            datasets
        """
        if type not in ['train', 'validation']:
            raise ValueError("Invalid type: must be 'train' or 'validation'")
        
        if type not in self.datasets:
                raise ValueError('--do_'+type+' requires a '+type+' dataset')
        
        datasets = self.process_data('pre')(self.datasets, type)

        return datasets
