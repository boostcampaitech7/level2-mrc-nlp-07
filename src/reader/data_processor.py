from abc import ABC
from abc import abstractmethod

from transformers import BatchEncoding


class DataProcessor(ABC):
    @classmethod
    @abstractmethod
    def process(cls, examples: BatchEncoding) -> BatchEncoding:
        pass


class DataPreProcessor(DataProcessor):
    @classmethod
    def process(cls, examples: BatchEncoding) -> BatchEncoding:
        """적절한 docstring 부탁드립니다.

        Args:
            examples (BatchEncoding): _description_

        Returns:
            BatchEncoding: _description_
        """
        print('Pre-processing...')
        return examples


class DataPostProcessor(DataProcessor):
    @classmethod
    def process(cls, examples: BatchEncoding) -> BatchEncoding:
        """적절한 docstring 부탁드립니다.

        Args:
            examples (BatchEncoding): _description_

        Returns:
            BatchEncoding: _description_
        """
        print('Post-processing...')
        return examples


if __name__ == '__main__':
    data = BatchEncoding(
        {
            'input_ids': [101, 102, 103],
            'attention_mask': [1, 1, 1],
        },
    )

    preprocessed_data = DataPreProcessor.process(data)
    print(preprocessed_data)

    postprocessed_data = DataPostProcessor.process(data)
    print(postprocessed_data)
