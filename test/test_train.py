import os
import sys
from unittest import TestCase
from unittest.mock import MagicMock
from unittest.mock import patch

sys.path.append('/data/ephemeral/home/level2-mrc-nlp-07/src')
from train import main


class TestMainFunction(TestCase):
    @patch('src.arguments.ModelArguments')  # ModelArguments의 정확한 경로
    # DataTrainingArguments의 정확한 경로
    @patch('src.arguments.DataTrainingArguments')
    @patch('transformers.TrainingArguments')  # transformers 모듈에서 가져옴
    @patch('src.reader.model.Reader')  # Reader의 경로
    @patch('src.reader.data_handler.DataHandler')  # DataHandler의 경로
    @patch('src.reader.evaluation.Evaluation')  # Evaluation의 경로
    @patch('src.reader.monitoring_tool.MonitoringTool')  # MonitoringTool의 경로
    @patch('transformers.Trainer')  # transformers 모듈에서 가져옴
    def test_main(
        self, mock_trainer, mock_monitoring_tool, mock_evaluation,
        mock_data_handler, mock_reader, mock_training_args,
        mock_data_training_args, mock_model_args,
    ):
        # Mock 객체 설정
        mock_model_args.return_value = MagicMock()
        mock_data_training_args.return_value = MagicMock()
        mock_training_args.return_value = MagicMock()

        # Mock Reader, DataHandler, Evaluation, MonitoringTool 및 Trainer 설정
        mock_reader_instance = MagicMock()
        mock_reader.return_value = mock_reader_instance

        mock_data_handler_instance = MagicMock()
        mock_data_handler.return_value = mock_data_handler_instance

        mock_eval_instance = MagicMock()
        mock_evaluation.return_value = mock_eval_instance

        mock_monitor_tool_instance = MagicMock()
        mock_monitoring_tool.return_value = mock_monitor_tool_instance

        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        # Mock 메서드 설정
        mock_data_handler_instance.load_data.side_effect = [
            {'data': 'train_data'}, {'data': 'eval_data'},
        ]

        # main 함수 실행
        main()

        # 각 객체의 메서드가 올바르게 호출되었는지 검증
        mock_reader.assert_called_once_with(mock_model_args.return_value)
        mock_reader_instance.load_model.assert_called_once()
        mock_reader_instance.load_tokenizer.assert_called_once()

        mock_data_handler.assert_called_once_with(
            mock_data_training_args.return_value,
        )
        mock_data_handler_instance.load_data.assert_any_call()

        mock_monitor_tool_instance.start_monitoring.assert_called_once()

        mock_trainer.assert_called_once()
        mock_trainer_instance.train.assert_called_once()
        mock_trainer_instance.evaluate.assert_called_once()

        mock_monitor_tool_instance.stop_monitoring.assert_called_once()


# 테스트 실행
if __name__ == '__main__':
    from unittest import main as unittest_main
    unittest_main()
