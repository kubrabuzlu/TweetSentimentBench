import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def get_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def compare_models_results(results_dict, save_path="model_comparison.csv"):
    """
    results_dict: Dict[model_name, Tuple[y_true, y_pred]]
    """
    rows = []
    for model_name, (y_true, y_pred) in results_dict.items():
        metrics = get_metrics(y_true, y_pred)
        metrics["model"] = model_name
        rows.append(metrics)

    df = pd.DataFrame(rows)
    df = df[["model", "accuracy", "precision", "recall", "f1_score"]]
    df.to_csv(save_path, index=False)
    print("\nModel Comparison Table:")
    print(df)
