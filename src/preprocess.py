import re
import string
from typing import List
import pandas as pd
from nltk.corpus import stopwords

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