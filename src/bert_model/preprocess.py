import pandas as pd
from typing import Tuple, Dict
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from src.dataloader import load_data


def encode_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int], Dict[int, str]]:
    """
    Encodes sentiment labels to integers.
    Args:
        df (pd.DataFrame): DataFrame with sentiment labels.
    Returns:
        Tuple: Updated DataFrame, label2id, id2label
    """
    label2id = {'negative': 0, 'neutral': 1, 'positive': 2}
    id2label = {v: k for k, v in label2id.items()}
    df['label'] = df['sentiment'].map(label2id)
    return df, label2id, id2label


def prepare_datasets_from_files(train_path: str, test_path: str) -> Tuple[DatasetDict, Dict[str, int], Dict[int, str]]:
    """
    Loads train and test CSVs and prepares a Hugging Face DatasetDict.
    Args:
        train_path (str): Path to training CSV file.
        test_path (str): Path to test CSV file.
    Returns:
        Tuple: DatasetDict with 'train' and 'test', label2id, id2label mappings.
    """
    train_df = load_data(train_path)
    test_df = load_data(test_path)

    train_df, label2id, id2label = encode_labels(train_df)
    test_df['label'] = test_df['sentiment'].map(label2id)

    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df[['tweet', 'label']]),
        'test': Dataset.from_pandas(test_df[['tweet', 'label']])
    })

    return dataset, label2id, id2label


def tokenize_dataset(dataset: DatasetDict, tokenizer: AutoTokenizer) -> DatasetDict:
    """
    Tokenizes the dataset using a given tokenizer.
    Args:
        dataset (DatasetDict): Hugging Face dataset.
        tokenizer (AutoTokenizer): Tokenizer object.
    Returns:
        DatasetDict: Tokenized dataset.
    """
    return dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding=True), batched=True)



