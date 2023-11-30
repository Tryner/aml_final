from typing import Callable
from random import shuffle

from datasets import Dataset, DatasetDict, concatenate_datasets
from setfit import SetFitModel, SetFitTrainer, sample_dataset

from torch.distributions import Categorical

from train.train_config import TrainConfig
from train.active_learning_config import ActiveLearningConfig
from data.dataset_config import DatasetConfig

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
        num_samples = active_learning_config.samples_per_cycle // dataset_config.num_classes
        subset = sample_dataset(train_dataset, label_column="label", num_samples=num_samples, seed=active_learning_config.seed)
    elif active_learning_config.initial_sample=="random":
        subset = train_dataset.shuffle(seed=active_learning_config.seed).select(range(active_learning_config.samples_per_cycle))
    else:
        raise ValueError("Not supported for initial sampling: " + active_learning_config.initial_sample)

    trainer = run_training(model_init=model_init, train_dataset=subset, eval_dataset=eval_dataset, train_config=train_config)
    print("Accuracy: " + str(trainer.evaluate()))
    for _ in range(active_learning_config.active_learning_cycles):
        sentences = [s for s in set(train_dataset["sentence"]) if s not in subset]
        sentences = select_sentences(sentences=sentences, trainer=trainer, active_learning_config=active_learning_config)
        subset = concatenate_datasets([subset, label_sentences(sentences, train_dataset)])

        trainer = run_training(model_init=model_init, train_dataset=subset, eval_dataset=eval_dataset, train_config=train_config)
        print("Accuracy: " + str(trainer.evaluate()))

    return trainer

def select_sentences(sentences: list[str], trainer: SetFitTrainer, active_learning_config: ActiveLearningConfig) -> list[str]:
    strategy = active_learning_config.sampling_strategy
    n_samples = active_learning_config.samples_per_cycle
    sentences = sentences[:active_learning_config.unlabeled_samples] #reduce computational cost
    if  strategy == "random":
        shuffle(sentences)
        return sentences[:n_samples]
    elif strategy == "max_entropy":
        probs = trainer.model.predict_proba(sentences)
        entropy = Categorical(probs=probs).entropy()
        sorted_sentences = sorted(zip(entropy, sentences), reverse=True)
        return [s for e, s in sorted_sentences][:n_samples]
    else:
        raise ValueError("Not a valid sampling_strategy: " + strategy)
    
def label_sentences(sentences: list[str], train_dataset: Dataset) -> Dataset:
    train_dataset = train_dataset.shuffle()
    indices = [train_dataset["sentence"].index(s) for s in sentences] 
    return train_dataset.select(indices)

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



