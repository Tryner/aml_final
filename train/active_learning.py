from typing import Callable
from random import shuffle

from datasets import Dataset, concatenate_datasets
from setfit import SetFitModel, Trainer, sample_dataset, TrainingArguments

from torch.distributions import Categorical

from train.active_learning_config import ActiveLearningConfig
from data.dataset_config import DatasetConfig
import time
class ActiveTrainer:

    def __init__(
            self,
            model_init: Callable[[], SetFitModel], 
            full_train_dataset: Dataset, 
            train_args: TrainingArguments,
            active_learning_config: ActiveLearningConfig,
            dataset_config: DatasetConfig,
            eval_dataset: Dataset = None,
            initial_train_subset: Dataset = None,
            after_train_callback: Callable[[Trainer], None] = None,
            dataset_callback: Callable[[Dataset], None] = None,
            ) -> None:
        self.model_init = model_init
        self.full_train_dataset = full_train_dataset
        self.train_args = train_args
        self.active_learning_config = active_learning_config
        self.dataset_config = dataset_config
        self.eval_dataset = eval_dataset
        self.after_train_callback = after_train_callback
        self.dataset_callback = dataset_callback

        self.train_subset = initial_train_subset
        if initial_train_subset is None:
            self.train_subset = self.create_initial_train_subset()
    
    def create_initial_train_subset(self) -> Dataset:
        samples_per_cycle = self.active_learning_config.samples_per_cycle
        seed = self.active_learning_config.seed
        label_column = self.dataset_config.label_column
        if self.active_learning_config.initial_sample=="balanced":
            assert samples_per_cycle % self.dataset_config.num_classes == 0
            num_samples = samples_per_cycle // self.dataset_config.num_classes
            train_subset = sample_dataset(self.full_train_dataset, label_column=label_column, num_samples=num_samples, seed=seed)
        elif self.active_learning_config.initial_sample=="random":
            train_subset = self.full_train_dataset.shuffle(seed=seed).select(range(samples_per_cycle))
        else:
            raise ValueError("Not supported for initial sampling: " + self.active_learning_config.initial_sample)
        train_subset = train_subset.cast_column(label_column, self.full_train_dataset.features[label_column])
        return train_subset

    def train(self) -> Trainer: 
        trainer = self.run_training()
        start_total_time = time.time()  # Misura l'inizio dell'addestramento
        for _ in range(self.active_learning_config.active_learning_cycles):
            sentences = self.select_sentences(trainer.model)
            labeled_sentences: Dataset = label_sentences(sentences, labeled_dataset=self.full_train_dataset, text_column=self.dataset_config.text_column)
            self.train_subset = concatenate_datasets([self.train_subset, labeled_sentences])
            self.dataset_callback(self.train_subset)
            trainer = self.run_training()
        end_total_time = time.time()  # Misura la fine dell'addestramento
        total_time = end_total_time - start_total_time
        print(f"Total time taken for training: {total_time} seconds")

        return trainer

    def select_sentences(self, model: SetFitModel) -> list[str]:
        text_column = self.dataset_config.text_column
        sentences = set(self.full_train_dataset[text_column])  # remove duplicates
        labeled_sentences = set(self.train_subset[text_column])  # already labeled sentences
        unlabeled_sentences = list(sentences.difference(labeled_sentences))

        print(f"Total sentences: {len(sentences)}")
        print(f"Labeled sentences: {len(labeled_sentences)}")
        print(f"Unlabeled sentences: {len(unlabeled_sentences)}")

        strategy = self.active_learning_config.sampling_strategy
        n_samples = self.active_learning_config.samples_per_cycle
        unlabeled_sentences = unlabeled_sentences[:self.active_learning_config.unlabeled_samples]  # reduce computational cost

        if strategy == "random":
            shuffle(unlabeled_sentences)
            selected_sentences = unlabeled_sentences[:n_samples]
        elif strategy == "max_entropy":
            probs = model.predict_proba(unlabeled_sentences)
            entropy = Categorical(probs=probs).entropy()
            sorted_sentences = sorted(zip(entropy, unlabeled_sentences), reverse=True)
            selected_sentences = [s for e, s in sorted_sentences][:n_samples]
        else:
            raise ValueError("Not a valid sampling_strategy: " + strategy)

        print(f"Selected sentences: {selected_sentences}")
        return selected_sentences


    def run_training(self) -> Trainer:
        trainer = Trainer(
            model_init=self.model_init,
            train_dataset=self.train_subset,
            eval_dataset=self.eval_dataset,
            args=self.train_args,
            metric=self.active_learning_config.metric,
            column_mapping={self.dataset_config.text_column: "text", self.dataset_config.label_column: "label"},
        )
        trainer.train()
        if self.after_train_callback: self.after_train_callback(trainer)
        return trainer
def label_sentences(sentences: list[str], labeled_dataset: Dataset, text_column: str="sentence") -> Dataset:
    labeled_dataset = labeled_dataset.shuffle()
    indices = [labeled_dataset[text_column].index(s) for s in sentences]

    # Visualizza le frasi etichettate
    print("Frasi etichettate:")
    for idx in indices:
        row = labeled_dataset[idx]
        print(f"{idx}. {row[text_column]} - Etichetta: {row['label']}")

    return labeled_dataset.select(indices)