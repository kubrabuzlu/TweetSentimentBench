from configparser import ConfigParser
from src.dataloader import load_data
from src.ml_models.preprocess import preprocess_data
from src.lstm_model.preprocess import tokenize_and_pad
from src.bert_model.preprocess import prepare_datasets_from_files
from src.ml_models.vectorizer import vectorize_data
from src.ml_models.ml_models import get_ml_models
from src.lstm_model.train_lstm import train_and_evaluate_lstm
from src.bert_model import train_model
from src.bert_model.compute_metrics import evaluate_model
from src.visualization import plot_confusion_matrix
from src.metrics import compare_models_results


def main():

    config = ConfigParser()
    config.read("config.ini")

    train_path = config["DATA"]["TRAIN_DIR"]
    test_path = config["DATA"]["TEST_DIR"]

    df_train = load_data(train_path)
    df_test = load_data(test_path)

    # ML Preprocess
    df_train = preprocess_data(df_train)
    df_test = preprocess_data(df_test)

    X_train = df_train['clean_tweet']
    y_train = df_train['sentiment']
    X_test = df_test['clean_tweet']
    y_test = df_test['sentiment']
    labels = sorted(y_train.unique())

    # ML Models
    X_train_vec, X_test_vec, _ = vectorize_data(X_train, X_test, y_train)
    ml_predictions_dict = get_ml_models(X_train_vec, y_train, X_test_vec)

    results = {}
    for model_name, y_pred in ml_predictions_dict.items():
        plot_confusion_matrix(y_test, y_pred, labels, model_name)
        results[model_name] = (y_test, y_pred)

    # LSTM
    lstm_params = {
        "embedding_dim": int(config["LSTM"]["embedding_dim"]),
        "lstm_units": int(config["LSTM"]["lstm_units"]),
        "dropout": float(config["LSTM"]["dropout"]),
        "recurrent_dropout": float(config["LSTM"]["recurrent_dropout"]),
        "dense_units": int(config["LSTM"]["dense_units"]),
        "dense_dropout": float(config["LSTM"]["dense_dropout"]),
        "epochs": int(config["LSTM"]["epochs"]),
        "batch_size": int(config["LSTM"]["batch_size"]),
    }
    X_train_pad, X_test_pad, lstm_tokenizer = tokenize_and_pad(X_train, X_test)
    lstm_preds = train_and_evaluate_lstm(X_train_pad, X_test_pad, y_train, y_test, lstm_tokenizer, lstm_params)
    plot_confusion_matrix(y_test, lstm_preds, labels, "LSTM")
    results["LSTM"] = (y_test, lstm_preds)

    # BERT
    bert_params = {
        "model_name": config["BERT"]["model_name"],
        "epochs": int(config["BERT"]["epochs"]),
        "batch_size": int(config["BERT"]["batch_size"]),
        "learning_rate": float(config["BERT"]["learning_rate"]),
    }
    dataset, label2id, id2label = prepare_datasets_from_files(train_path, test_path)
    trainer, tokenized_dataset = train_model(dataset=dataset, label2id=label2id, id2label=id2label, **bert_params)
    bert_preds = evaluate_model(trainer, tokenized_dataset, tokenized_dataset["test"]["label"])
    bert_y_true = tokenized_dataset["test"]["label"]
    bert_y_pred_str = [id2label[p] for p in bert_preds]
    bert_y_true_str = [id2label[t] for t in bert_y_true]

    plot_confusion_matrix(bert_y_true_str, bert_y_pred_str, labels, "BERT")
    results["BERT"] = (bert_y_true_str, bert_y_pred_str)

    # Compare all models
    compare_models_results(results)


if __name__ == "__main__":

    main()

