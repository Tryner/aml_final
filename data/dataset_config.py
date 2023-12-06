from dataclasses import dataclass

#sst2
@dataclass
class DatasetConfig:
    text_column: str = "sentence"
    label_column: str = "label"
    num_classes: int = 2