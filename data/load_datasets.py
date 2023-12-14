import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset,DatasetDict

class DatasetConfig:
    text_column: str = "sentence"
    label_column: str = "label"
    num_classes: int = 2
    
def load(choice):
    match choice:
        case 1:
            name="sst2"
            d = load_dataset(name)
        case 2:
            name="french_twitter_reduced"
            d = load_french_dataset(name)
        case 3:
            name="twitter_humor"
            d =load_spanish_dataset(name)
        case 4:
            name="datd_train"
            d = load_indonesian_dataset(name)
        case _:
            print('wrong selection') 
    return name, d
    
def select_dataset():
    print("Select dataset:")
    print("1. sst2")
    print("2. french")
    print("3. spanish")
    print("4. indonesian")
    choice = input("Enter choice (1, 2, 3, or 4): ")
    return int(choice)

    
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

def label_row(row):
    if row["prejudice_woman"] == 1:
        return 0
    if row["prejudice_lgbtiq"] == 1:
        return 1
    if row["prejudice_inmigrant_race"] == 1:
        return 2
    if row["gordofobia"] == 1:
        return 3
    raise ValueError("label missing!")

def load_spanish_dataset(name):
    df=pd.read_csv(f'data/raw_datasets/{name}.csv')

    del df["index"]
    del df["mean_prejudice"]
    df["sum"] = df.loc[:, ['prejudice_woman', 'prejudice_lgbtiq', 'prejudice_inmigrant_race', 'gordofobia']].sum(axis=1)
    df = df.loc[df['sum'] == 1]
    del df["sum"]
    df["label"] = [label_row(row) for _, row in df.iterrows()]

    X = df['tweet']
    y = df.drop('tweet', axis=1)

    # Split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create DataFrames for training and validation
    tset = Dataset.from_pandas(pd.concat([X_train, y_train], axis=1))
    vset = Dataset.from_pandas(pd.concat([X_valid, y_valid], axis=1))
    dataset_dict = DatasetDict({'train': tset, 'validation': vset})
    return dataset_dict



def load_indonesian_dataset(name):
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