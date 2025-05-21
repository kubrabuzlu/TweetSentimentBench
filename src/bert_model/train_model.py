from datasets import DatasetDict
from typing import Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

from preprocess import tokenize_dataset
from compute_metrics import compute_metrics


def train_model(
    dataset: DatasetDict,
    model_name: str,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    num_train_epochs: int,
    batch_size: int,
    learning_rate: float,
    max_len: int
) -> Trainer:
    """
    Trains a BERT model on the dataset using provided hyperparameters.

    Args:
        dataset (DatasetDict): Raw Hugging Face dataset.
        model_name (str): Name of the pretrained BERT model.
        label2id (Dict[str, int]): Label to ID mapping.
        id2label (Dict[int, str]): ID to label mapping.
        num_train_epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        learning_rate (float): Learning rate for optimizer.
        max_len (int): Maximum sequence length.

    Returns:
        Trainer: Trained Hugging Face Trainer object.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label
    )

    tokenized_dataset = tokenize_dataset(dataset, tokenizer, max_len=max_len)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer, tokenized_dataset

