from typing import Callable

from datasets import Dataset, DatasetDict
from setfit import SetFitModel, SetFitTrainer, sample_dataset

from .train_config import TrainConfig
from .active_learning_config import ActiveLearningConfig
from ..data.dataset_config import DatasetConfig

def active_train(
        model_init: Callable[[], SetFitModel], 
        dataset: DatasetDict, 
        train_config: TrainConfig, 
        active_learning_config: ActiveLearningConfig,
        dataset_config: DatasetConfig
        ) -> SetFitTrainer: 
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    #create initial training dataset
    if active_learning_config.initial_sample=="balanced":
        assert active_learning_config.samples_per_cycle % dataset_config.num_classes == 0
        subset = sample_dataset(train_dataset, label_column="label", num_samples=active_learning_config.samples_per_cycle // dataset_config.num_classes)
    elif active_learning_config.initial_sample=="random":
        subset = train_dataset.shuffle(seed=active_learning_config.seed).select(range(active_learning_config.samples_per_cycle))
    else:
        raise ValueError("Not supported for initial sampling!")

    trainer = run_training(model_init=model_init, train_dataset=subset, eval_dataset=eval_dataset, train_config=train_config)
    for _ in range(active_learning_config.active_learning_cycles):
        
        subset = subset #TODO add examples
        trainer = run_training(model_init=model_init, train_dataset=subset, eval_dataset=eval_dataset, train_config=train_config)
    return trainer


def run_training(
        model_init: Callable[[], SetFitModel], 
        train_dataset: Dataset, 
        eval_dataset: Dataset, 
        train_config: TrainConfig
        ) -> SetFitTrainer:
    trainer = SetFitTrainer(
        model_init=model_init,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
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
    return trainer



