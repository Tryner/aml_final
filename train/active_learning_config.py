from dataclasses import dataclass

class ActiveLearningConfig:
    active_learning_cycles: int = 3
    samples_per_cycle: int = 25