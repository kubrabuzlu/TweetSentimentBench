import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import Tuple


def tokenize_and_pad(X_train: pd.Series,
                     X_test: pd.Series,
                     max_len: int = 100,
                     num_words: int = 10000) -> Tuple[np.ndarray, np.ndarray, Tokenizer]:
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

