from typing import Callable
import random
from collections import Counter
import gc

from datasets import Dataset, concatenate_datasets
from setfit import SetFitModel, Trainer, sample_dataset, TrainingArguments
import torch
from torch.distributions import Categorical

from train.active_learning_config import ActiveLearningConfig
from data.dataset_config import DatasetConfig

def create_random_subset(dataset: Dataset, dataset_config: DatasetConfig, num_samples: int, num_balanced_samples: int = 1, seed: int = 42):
    """Randomly samples a subset from a larger dataset."""
    balanced_subset = sample_dataset(dataset, dataset_config.label_column, num_balanced_samples, seed=seed) # ensure that we have at least one example per class
    return add_random_samples(
        dataset=dataset, subset=balanced_subset, num_samples=num_samples, 
        dataset_config=dataset_config, seed=seed
        )

def add_random_samples(dataset: Dataset, subset: Dataset, num_samples: int, dataset_config: DatasetConfig, seed: int):
    text_column = dataset_config.text_column
    label_column = dataset_config.label_column
    random_samples = dataset.shuffle(seed=seed).select(range(num_samples))
    random_samples = random_samples.filter(lambda e: e[text_column] not in subset[text_column]) # remove duplicates
    random_samples = random_samples.select(range(num_samples-len(subset))) # len(random) + len(subset) = num_samples
    subset = subset.cast_column(label_column, random_samples.features[label_column]) # sample_dataset sometimes looses some information, preventing concat
    return concatenate_datasets([random_samples, subset])


class ActiveTrainer:
    """Trainer for active learning"""

    def __init__(
            self,
            full_train_dataset: Dataset, 
            train_args: TrainingArguments,
            active_learning_config: ActiveLearningConfig,
            dataset_config: DatasetConfig,
            model_init: Callable[[], SetFitModel] | None = None, 
            eval_dataset: Dataset = None,
            initial_train_subset: Dataset = None,
            metric: Callable | str= "accuracy", 
            after_train_callback: Callable[[Trainer, Dataset, int], None] = None,
            run_id: int = -1,
            final_model_train_args: TrainingArguments = None,
            ) -> None:
        self.full_train_dataset = full_train_dataset
        self.train_args = train_args
        self.active_learning_config = active_learning_config
        self.dataset_config = dataset_config
        self.eval_dataset = eval_dataset
        self.metric = metric
        self.after_train_callback = after_train_callback
        self.run_id = run_id

        self.random = random.Random(x=active_learning_config.random_seed)

        self.model_init = model_init
        if model_init is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model_init = lambda: SetFitModel.from_pretrained(
                active_learning_config.model_name, use_differentiable_head=True, head_params={"out_features": dataset_config.num_classes}
                ).to(device)

        self.final_model_train_args = final_model_train_args
        if final_model_train_args is None:
            self.final_model_train_args = train_args

        self.train_subset = initial_train_subset
        if initial_train_subset is None:
            self.train_subset = self.create_initial_train_subset()
    
    def create_initial_train_subset(self) -> Dataset:
        n_samples = self.active_learning_config.samples_per_cycle
        seed = self.active_learning_config.random_seed
        label_column = self.dataset_config.label_column
        if self.active_learning_config.initial_sample=="balanced":
            assert n_samples % self.dataset_config.num_classes == 0
            num_samples = n_samples // self.dataset_config.num_classes
            train_subset = sample_dataset(self.full_train_dataset, label_column=label_column, num_samples=num_samples, seed=seed)
        elif self.active_learning_config.initial_sample=="random":
            train_subset = create_random_subset(self.full_train_dataset, dataset_config=self.dataset_config, num_samples=n_samples, seed=seed)
        else:
            raise ValueError("Not supported for initial sampling: " + self.active_learning_config.initial_sample)
        train_subset = train_subset.cast_column(label_column, self.full_train_dataset.features[label_column])
        return train_subset

    def train(self) -> Trainer: 
        """Main Active learning loop."""
        for _ in range(self.active_learning_config.active_learning_cycles):
            model = self.run_training().model
            sentences = self.select_sentences(model)
            labeled_sentences = label_sentences(sentences, labeled_dataset=self.full_train_dataset, text_column=self.dataset_config.text_column)
            self.train_subset = concatenate_datasets([self.train_subset, labeled_sentences])
        trainer = self.run_training(final_model=True)
        return trainer

    def select_sentences(self, model: SetFitModel) -> list[str]:
        """Selects the sentences to be labeled for active learning"""
        text_column = self.dataset_config.text_column
        sentences = set(self.full_train_dataset[text_column]) # remove duplicates
        sentences = list(sentences.difference(set(self.train_subset[text_column]))) # remove already labeled sentences
        strategy = self.active_learning_config.active_sampling_strategy
        n_samples = self.active_learning_config.samples_per_cycle
        self.random.shuffle(sentences) #ensure that we see different sentences every cycle
        sentences = sentences[:self.active_learning_config.unlabeled_samples] #reduce computational cost
        
        if  strategy == "random":
            return sentences[:n_samples] # we don't need to run inference to select random sentences
        
        probs = model.predict_proba(sentences).cpu()
        if self.active_learning_config.balancing_factor is not None:
            size = len(self.train_subset)
            counter = Counter(self.train_subset[self.dataset_config.label_column])
            dist = torch.tensor([counter[label] for label in range(self.dataset_config.num_classes)]) / size #relative label distribution
            dist_weight = self.active_learning_config.balancing_factor
            probs = ((1- dist_weight) * probs) + (dist_weight * dist) #weighted average of predicted probabilities and label distribution
        
        if strategy == "max_entropy":
            #selects the sentences with the highest entropy
            entropy = Categorical(probs=probs).entropy()
            sorted_sentences = sorted(zip(entropy, sentences), reverse=True)
            return [s for e, s in sorted_sentences][:n_samples]
        
        raise ValueError("Not a valid sampling_strategy: " + strategy)


    def run_training(self, final_model: bool = False) -> Trainer:
        """Trains a SetFit model"""
        gc.collect() #free up vram
        args = self.train_args
        if final_model: args = self.final_model_train_args
        trainer = Trainer(
            model_init=self.model_init,
            train_dataset=self.train_subset,
            eval_dataset=self.eval_dataset,
            args=args,
            metric=self.metric,
            column_mapping={self.dataset_config.text_column: "text", self.dataset_config.label_column: "label"},
        )
        trainer.train()
        if self.after_train_callback: self.after_train_callback(trainer, self.train_subset, self.run_id)
        return trainer

#Also possible to ask for user input here, for "real" active learning
def label_sentences(sentences: list[str], labeled_dataset: Dataset, text_column: str="sentence") -> Dataset:
    """Labels Sentences using a large, already labeled dataset as an oracle. Assumes that all sentences are contained in the labeld dataset."""
    labeled_dataset = labeled_dataset.shuffle()
    indices = [labeled_dataset[text_column].index(s) for s in sentences] 
    return labeled_dataset.select(indices)
