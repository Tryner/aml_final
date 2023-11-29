from datasets import Dataset, DatasetDict
from setfit import SetFitModel, SetFitTrainer

from .train_config import TrainConfig
from .active_learning_config import ActiveLearningConfig

def active_train(model: SetFitModel, dataset: Dataset, train_config: TrainConfig, active_learning_config: ActiveLearningConfig):
    pass


def run_training(model: SetFitModel, train_dataset: Dataset, train_config: TrainConfig):
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_dataset,
        loss_class=train_config.loss_class,
        metric=train_config.metric,
        batch_size=train_config.batch_size,
        num_iterations=train_config.num_iterations,
        num_epochs=train_config.num_epochs,
        column_mapping={"sentence": "text", "label": "label"} # Map dataset columns to text/label expected by trainer
    )
    trainer.freeze() # Freeze the head
    trainer.train() # Train only the body
    # Unfreeze the head and freeze the body -> head-only training
    #trainer.unfreeze(keep_body_frozen=True)
    # OR: Unfreeze the head and unfreeze the body -> end-to-end training
    trainer.unfreeze(keep_body_frozen=False)
    trainer.train(
        num_epochs=10, 
        batch_size=8,
        body_learning_rate=1e-5, 
        learning_rate=1e-2,
        l2_weight=0.0,
    ) # add to TrainConfig
    return trainer.model



