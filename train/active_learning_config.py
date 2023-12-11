from typing import Literal
from dataclasses import dataclass

from datasets import Dataset

@dataclass
class ActiveLearningConfig:
    active_learning_cycles: int = 3
    samples_per_cycle: int = 8
    initial_sample: Literal["balanced", "random"] = "balanced"
    sampling_strategy: Literal["random", "max_entropy"] = "max_entropy"
    seed: int = 42
    unlabeled_samples: int = 160
    
    model_name: str = "thenlper/gte-small"
