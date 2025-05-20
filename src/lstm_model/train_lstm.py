import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from typing import List

from lstm_model import build_lstm_model


def train_and_evaluate_lstm(X_train_pad: np.ndarray, X_test_pad: np.ndarray,
                            y_train: pd.Series, y_test: pd.Series, tokenizer: Tokenizer) -> List:
    """
    Train and evaluate an LSTM model.

    Args:
        X_train_pad: Padded training sequences.
        X_test_pad: Padded testing sequences.
        y_train: Training labels.
        y_test: Testing labels.
        tokenizer: Fitted tokenizer.
    """
    label_map = {'positive': 0, 'neutral': 1, 'negative': 2}
    y_train_cat = to_categorical(y_train.map(label_map))
    y_test_cat = to_categorical(y_test.map(label_map))

    model = build_lstm_model(input_length=X_train_pad.shape[1], vocab_size=len(tokenizer.word_index) + 1)

    model.fit(X_train_pad, y_train_cat, epochs=5, batch_size=64, validation_data=(X_test_pad, y_test_cat), verbose=2)

    y_pred = model.predict(X_test_pad)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test_cat, axis=1)

    print("\nLSTM Classification Report:\n")
    report = classification_report(y_test_classes, y_pred_classes, target_names=['positive', 'neutral', 'negative'])
    print(report)

    id2label = {0: 'positive', 1: 'neutral', 2: 'negative'}
    y_pred_str = [id2label[i] for i in y_pred_classes]

    return y_pred_str

