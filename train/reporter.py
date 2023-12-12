from dataclasses import asdict
from collections import Counter

from datasets import Dataset
from setfit import Trainer

from train.active_learning_config import ActiveLearningConfig

def write_csv(file_name: str, data: list[str], delimmiter: str = ";"):
    data = map(str, data) # ensure that we have strings
    line = delimmiter.join(data) + "\n"
    with open(file_name, mode="a") as f:
        f.write(line)

def write_dict(file_name:str, column_names: list[str], values: dict[str, str]):
    data = [values[key] for key in column_names]
    write_csv(file_name, data)

def describe_dataset(dataset: Dataset, label_column: str = "label"):
    size = len(dataset)
    counter = Counter(dataset[label_column])
    description = {"dataset_size": size}
    for label, amount in counter.items():
        description.update({
            "abs_" + str(label): amount,
            "rel_" + str(label): f"{amount/size:.2f}"
        })
    return description

    
class Reporter:
    def __init__(
            self, 
            file_name: str, 
            report_train_args: bool = True, 
            column_names: list[str] = None, 
            label_column: str = "label"
            ) -> None:
        self.file_name = file_name
        self.column_names = column_names
        self.report_train_args = report_train_args
        self.label_column = label_column
        
    # def report(self, trainer: Trainer = None, dataset: Dataset = None, active_learning_config: ActiveLearningConfig = None, **other_params):    
    def report(self, elapsed_time, trainer: Trainer = None, dataset: Dataset = None, active_learning_config: ActiveLearningConfig = None, **other_params):
        if trainer:
            metrics = trainer.evaluate()
            metrics = {k: f"{v:.3f}" for k, v in metrics.items()}
        else:
            metrics = {}

        if trainer and self.report_train_args:
            train_args = asdict(trainer.args)
        else:
            train_args = {}

        if dataset:
            dataset_description = describe_dataset(dataset=dataset, label_column=self.label_column)
        else:
            dataset_description ={}

        if active_learning_config:
            active_learning_config = asdict(active_learning_config)
        else:
            active_learning_config = {}
        
        if self.column_names is None:
            self.column_names = list(other_params.keys()) + list(metrics.keys()) + list(dataset_description.keys()) + list(active_learning_config.keys()) + list(train_args.keys())
            duplicates = [column for column, amount in Counter(self.column_names).items() if amount>1]
            if len(duplicates) > 0:
                raise ValueError("Duplicated keys: " + str(duplicates))
            write_csv(self.file_name, self.column_names) #first call, so write column names
            
        # if elapsed_time['time']:
        #     elapsed_time=asdict{elapsed_time}
        # else:
        #     elapsed_time = {}

        data = other_params | metrics | dataset_description | active_learning_config | train_args | elapsed_time # dict Union
        # data = other_params | metrics | dataset_description | active_learning_config | train_args # dict Union
        write_dict(self.file_name, column_names=self.column_names, values=data)
