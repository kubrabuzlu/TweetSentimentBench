import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import Dict


def plot_confusion_matrices(y_test, predictions: Dict[str, list[str]]) -> None:
    """
    Plot confusion matrices for each model.

    Args:
        y_test: True labels.
        predictions (dict): Dictionary of model predictions.
    """
    fig, ax = plt.subplots(1, len(predictions), figsize=(18, 5))
    for i, (name, y_pred) in enumerate(predictions.items()):
        cm = confusion_matrix(y_test, y_pred, labels=["positive", "neutral", "negative"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["positive", "neutral", "negative"])
        disp.plot(ax=ax[i], cmap='Blues', values_format='d')
        ax[i].set_title(name)
    plt.tight_layout()
    plt.show()