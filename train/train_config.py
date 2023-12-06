from dataclasses import dataclass

from torch import nn
from sentence_transformers.losses import CosineSimilarityLoss

@dataclass
class TrainConfig:
    loss_class: nn.Module = CosineSimilarityLoss
    metric: str = "accuracy"
    contrastive_batch_size: int = 8
    num_iterations: int = 20

    head_epochs: int = 1
    head_batch_size: int = 8
    body_learning_rate: float = 1e-5
    head_learning_rate: float = 1e-2
    head_weight_decay: float = 0.0