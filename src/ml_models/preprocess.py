import re
import string
from typing import List
import pandas as pd
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

stop_words = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """
    Clean tweet text by removing URLs, mentions, hashtags, punctuation, numbers, and stopwords.

    Args:
        text (str): Raw tweet text.

    Returns:
        str: Cleaned tweet text.
    """
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess tweets by cleaning text and adding a new column with cleaned content.

    Args:
        df (pd.DataFrame): DataFrame with raw tweets.

    Returns:
        pd.DataFrame: Updated DataFrame with a 'clean_tweet' column.
    """
    df['clean_tweet'] = df['tweet'].apply(clean_text)
    return df


def tokenize_and_pad(X_train: pd.Series, X_test: pd.Series, max_len: int = 100, num_words: int = 10000) -> Tuple[np.ndarray, np.ndarray, Tokenizer]:
    """
    Tokenize and pad text sequences.

    Args:
        X_train (pd.Series): Training text data.
        X_test (pd.Series): Testing text data.
        max_len (int): Maximum sequence length.
        num_words (int): Maximum vocabulary size.

    Returns:
        Tuple of padded sequences and tokenizer.
    """
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

    return X_train_pad, X_test_pad, tokenizer
