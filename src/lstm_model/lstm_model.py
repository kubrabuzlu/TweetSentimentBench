from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout


def build_lstm_model(input_length: int, vocab_size: int, num_classes: int = 3) -> Sequential:
    """
    Build and return an LSTM model.

    Args:
        input_length (int): Length of input sequences.
        vocab_size (int): Size of vocabulary.
        num_classes (int): Number of sentiment classes.

    Returns:
        Compiled Keras model.
    """
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=input_length))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


