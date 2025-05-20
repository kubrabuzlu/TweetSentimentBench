import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import Dict


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import Dict, List


def plot_confusion_matrices(
    y_test: List[str],
    predictions: Dict[str, List[str]],
    labels: List[str],
    save_path: str = "confusion_matrix.png"
) -> None:
    """
    Plot and optionally save confusion matrices for each model.

    Args:
        y_test (list): True labels.
        predictions (dict): Model predictions. Key is model name.
        labels (list): Class labels in correct order.
        save_path (str, optional): File path to save the figure (e.g., 'conf_matrices.png').
    """
    fig, ax = plt.subplots(1, len(predictions), figsize=(6 * len(predictions), 5))
    if len(predictions) == 1:
        ax = [ax]

    for i, (name, y_pred) in enumerate(predictions.items()):
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(cm, display_labels=labels)
        disp.plot(ax=ax[i], cmap='Blues', values_format='d')
        ax[i].set_title(f"Model: {name}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
