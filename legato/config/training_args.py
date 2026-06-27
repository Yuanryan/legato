from dataclasses import dataclass, field
from typing import Union, Optional

@dataclass
class DataArguments:
    dataset_path: str
    mini_val_file: str = field(
        default=None,
        metadata={"help": "Path to the mini validation file. If not provided, the the whole validation set will be used."},
    )
    mini_test_file: str = field(
        default=None,
        metadata={"help": "Path to the mini test file. If not provided, the the whole test set will be used."},
    )
    dummy_data: bool = field(
        default=False,
        metadata={"help": "Use dummy data (32 items) for testing purposes."},
    )
    image_column: str = field(
        default="image",
        metadata={"help": "Dataset column to use as input image."},
    )
    transcription_column: str = field(
        default="transcription",
        metadata={"help": "Dataset column containing the ground-truth ABC transcription."},
    )
    predict_split: str = field(
        default="test",
        metadata={"help": "Dataset split to run --do_predict on (e.g. 'test', 'train')."},
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate the predict split to this many samples (useful for debugging)."},
    )

@dataclass
class ModelArguments:
    model_config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the model configuration file."},
    )
    pretrained_model: Optional[str] = field(
        default=None,
        metadata={"help": "Set the path to load a pretrained model before training or evaluation."},
    )
