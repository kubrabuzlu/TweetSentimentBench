from typing import Dict, List, Any
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


def get_ml_models(X_train_vec, X_test_vec, y_train, y_test) -> Dict[str, List[Any]]:
    """
    Train multiple ML models and evaluate them using classification reports.

    Args:
        X_train_vec: Vectorized training features.
        X_test_vec: Vectorized testing features.
        y_train: Training labels.
        y_test: Testing labels.

    Returns:
        dict: Predictions of each model on the test set.
    """

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC(max_iter=5000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "KNN": KNeighborsClassifier()
    }
    predictions = {}

    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        predictions[name] = y_pred
        print(f"\n{name} Classification Report:\n")
        print(classification_report(y_test, y_pred))

    return predictions
