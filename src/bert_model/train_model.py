from datasets import DatasetDict
from typing import Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

from preprocess import tokenize_dataset
from compute_metrics import compute_metrics


def train_model(dataset: DatasetDict, model_name: str, label2id: Dict[str, int], id2label: Dict[int, str]) -> Trainer:
    """
    Trains a BERT model on the dataset.
    Args:
        dataset (DatasetDict): Tokenized dataset.
        model_name (str): Name of the BERT model.
        label2id (dict): Label to ID mapping.
        id2label (dict): ID to label mapping.
    Returns:
        Trainer: Trained Hugging Face Trainer object.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, label2id=label2id, id2label=id2label)

    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
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
    return trainer

