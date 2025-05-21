# Tweet Sentiment Analysis Project
A comparative sentiment analysis benchmark on tweets using classical Machine Learning models, Recurrent Neural Networks (LSTM), and Transformer-based models (BERT). The goal is to evaluate and compare model performance across traditional and deep learning techniques on the same tweet sentiment dataset.

---

## Features

- **Data Loading and Preprocessing:** Load and clean Twitter data for sentiment analysis.
- **Classical ML Models:** Utilize TF-IDF vectorization and train models like Naive Bayes, Logistic Regression, etc.
- **LSTM Model:** Use word embeddings and LSTM layers for deep learning based sentiment classification.
- **BERT Model:** Fine-tune a pre-trained Transformer-based BERT model for sentiment analysis.
- **Config-driven:** Model parameters and dataset paths are managed via a `config.ini` file.
- **Results Visualization:** Generate confusion matrices and compare model metrics visually.

---

## Project Structure
```
├── config.ini # Configuration file for parameters and data paths 
├── main.py # Main script to train, test, and compare models
├── src
│ ├── dataloader.py # Functions for data loading
│ ├── ml_models
│ │ ├── preprocess.py # Preprocessing for classical ML models
│ │ ├── vectorizer.py # TF-IDF vectorization
│ │ ├── ml_models.py # ML model training and prediction
│ ├── lstm_model
│ │ ├── preprocess.py # Tokenization and padding for LSTM
│ │ ├── train_lstm.py # LSTM training and evaluation
│ ├── bert_model
│ │ ├── preprocess.py # Data preparation for BERT
│ │ ├── train_model.py # BERT training script
│ │ ├── compute_metrics.py # Evaluation metrics for BERT
│ ├── visualization.py # Plotting confusion matrices and other visuals
│ ├── metrics.py # Compare performance metrics across models
├── requirements.txt # Required Python packages
└── README.md # This documentation file
```