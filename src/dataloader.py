from typing import Literal
import pandas as pd


def load_data(filepath: str, sentiments: list[Literal["positive", "neutral", "negative"]] = None) -> pd.DataFrame:
    """
    Load dataset and filter by sentiment.

    Args:
        filepath (str): CSV path.
        sentiments (list): List of valid sentiment labels to filter by.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    df = pd.read_csv(filepath)[['tweet', 'sentiment']]
    if sentiments:
        df = df[df['sentiment'].isin(sentiments)]
    return df.reset_index(drop=True)
