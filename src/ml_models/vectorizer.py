from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif


def vectorize_data(X_train: pd.Series,
                   X_test: pd.Series,
                   y_train: pd.Series,
                   k_features: int = 3000) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """
    Vectorize text data using TF-IDF and apply feature selection.

    Args:
        X_train (pd.Series): Training text data.
        X_test (pd.Series): Testing text data.
        y_train (pd.Series): Training labels.
        k_features (int): Number of best features to select (default is 3000).

    Returns:
        Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
            Vectorized and reduced training and test data, and the fitted vectorizer.
    """
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    selector = SelectKBest(score_func=mutual_info_classif, k=min(k_features, X_train_vec.shape[1]))
    X_train_sel = selector.fit_transform(X_train_vec, y_train)
    X_test_sel = selector.transform(X_test_vec)

    return X_train_sel, X_test_sel, vectorizer
