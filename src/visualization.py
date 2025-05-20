import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import List


def plot_confusion_matrix(
        y_true: List[str],
        y_pred: List[str],
        labels: List[str],
        model_name: str,
        save_dir: str = "confusion_matrices"
) -> None:
    """
    Plot and save a single confusion matrix.

    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        labels (list): All class labels.
        model_name (str): Name of the model (used in the plot title and filename).
        save_dir (str): Directory to save the plot.
    """
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format='d')
    ax.set_title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"confusion_matrix_{model_name}.png"), dpi=300)
    plt.close()
