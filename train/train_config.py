from dataclasses import dataclass

from torch import nn
from sentence_transformers.losses import CosineSimilarityLoss

@dataclass
class TrainConfig:
    loss_class: nn.Module = CosineSimilarityLoss
    metric = "accuracy"
    batch_size = 8
    num_iterations = 5
    num_epochs = 1