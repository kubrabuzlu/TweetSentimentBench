from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout


def build_lstm_model(input_length: int, vocab_size: int, num_classes: int = 3,
                     embedding_dim: int = 128,
                     lstm_units: int = 128,
                     lstm_dropout: float = 0.2,
                     recurrent_dropout: float = 0.2,
                     dense_units: int = 64,
                     dense_dropout: float = 0.3) -> Sequential:
    """
    Build and return an LSTM model with configurable parameters.

    Args:
        input_length (int): Length of input sequences.
        vocab_size (int): Size of vocabulary.
        num_classes (int): Number of sentiment classes.
        embedding_dim (int): Dimension of embedding layer.
        lstm_units (int): Number of units in LSTM layer.
        lstm_dropout (float): Dropout rate for LSTM layer.
        recurrent_dropout (float): Recurrent dropout for LSTM layer.
        dense_units (int): Units in the dense layer before output.
        dense_dropout (float): Dropout after dense layer.

    Returns:
        Compiled Keras model.
    """
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length))
    model.add(LSTM(lstm_units, dropout=lstm_dropout, recurrent_dropout=recurrent_dropout))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dense_dropout))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
