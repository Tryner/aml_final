from typing import Literal
from dataclasses import dataclass

from datasets import Dataset

class ActiveLearningConfig:
    active_learning_cycles: int = 3
    samples_per_cycle: int = 25
    initial_sample: Literal["balanced", "random"] = "balanced"
    seed = 42
