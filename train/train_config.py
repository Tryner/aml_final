from dataclasses import dataclass

from torch import nn
from sentence_transformers.losses import CosineSimilarityLoss

@dataclass
class TrainConfig:
    loss_class: nn.Module = CosineSimilarityLoss
    metric: str = "accuracy"
    batch_size: int = 8
    num_iterations: int = 20
    num_epochs: int = 1