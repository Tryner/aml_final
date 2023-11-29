from dataclasses import dataclass

#sst2
@dataclass
class DatasetConfig:
    column_mapping: dict[str, str] = {"sentence": "text", "label": "label"}
    num_classes: int = 2