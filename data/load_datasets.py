import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset,DatasetDict

class DatasetConfig:
    text_column: str = "sentence"
    label_column: str = "label"
    num_classes: int = 2
    
def load_french_dataset(name):

    df=pd.read_csv(f'data/raw_datasets/{name}.csv')

    df=df.loc[:,['text','label']]
    df=df.rename(columns={'text': DatasetConfig.text_column, 'label': DatasetConfig.label_column})
    X = df.drop('label', axis=1)  # Assuming 'label' is the name of your label column
    y = df['label']

    # Split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create DataFrames for training and validation
    tset = Dataset.from_pandas(pd.concat([X_train, y_train], axis=1))
    vset = Dataset.from_pandas(pd.concat([X_valid, y_valid], axis=1))
    dataset_dict = DatasetDict({'train': tset, 'validation': vset})
    return dataset_dict


def load_spanish_dataset(name):
    df=pd.read_csv(f'data/raw_datasets/{name}.csv')

    df=df.loc[:,['tweet','humor']]
    df=df.rename(columns={'tweet': DatasetConfig.text_column, 'humor': DatasetConfig.label_column})
    X = df.drop('label', axis=1)  # Assuming 'label' is the name of your label column
    y = df['label']

    # Split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create DataFrames for training and validation
    tset = Dataset.from_pandas(pd.concat([X_train, y_train], axis=1))
    vset = Dataset.from_pandas(pd.concat([X_valid, y_valid], axis=1))
    dataset_dict = DatasetDict({'train': tset, 'validation': vset})
    return dataset_dict