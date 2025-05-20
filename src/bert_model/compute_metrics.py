import numpy as np
import evaluate
from typing import Tuple, Dict
from sklearn.metrics import classification_report

def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Computes evaluation metrics from predictions and labels.
    Args:
        eval_pred (Tuple): Tuple of predictions and labels.
    Returns:
        Dict[str, float]: Accuracy and F1 score.
    """
    metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='weighted')
    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}


def evaluate_model(trainer: Trainer, dataset: DatasetDict, original_labels: List[int]) -> None:
    """
    Evaluates the model on the test set and prints a classification report.
    Args:
        trainer (Trainer): Hugging Face Trainer.
        dataset (DatasetDict): Tokenized dataset.
        original_labels (List[int]): Ground-truth labels for the test set.
    """
    predictions = trainer.predict(dataset["test"])
    pred_labels = np.argmax(predictions.predictions, axis=1)
    print(classification_report(original_labels, pred_labels, target_names=['negative', 'neutral', 'positive']))