from typing import Dict
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def get_ml_models() -> Dict[str, object]:
    """
    Return a dictionary of predefined machine learning models for text classification.

    Models included:
        - Logistic Regression
        - Multinomial Naive Bayes
        - Support Vector Machine (LinearSVC)
        - Decision Tree
        - Random Forest
        - KNN

    Returns:
        Dict[str, object]: Dictionary mapping model names to their sklearn instances.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "KNN": KNeighborsClassifier()
    }
