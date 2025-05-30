import numpy as np
from utils import sigmoid, tanh, softmax, relu, calculate_f1_macro
import pandas as pd
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
import os

class ManualEmbeddingLayer:
    def __init__(self, embedding_matrix):
        self.embedding_matrix = embedding_matrix

    def forward(self, X_batch):
        return self.embedding_matrix[X_batch]

class ManualDenseLayer:
    def __init__(self, weights, bias, activation_fn=None):
        self.weights = weights
        self.bias = bias
        self.activation_fn = activation_fn

    def forward(self, X_batch):
        output = np.dot(X_batch, self.weights) + self.bias
        if self.activation_fn:
            output = self.activation_fn(output)
        return output

class ManualRNNCell:
    def __init__(self, Wx, Wh, b, activation=tanh):
        self.Wx = Wx
        self.Wh = Wh
        self.b = b
        self.activation = activation

    def forward(self, xt, h_prev):
        h_next = self.activation(np.dot(xt, self.Wx) + np.dot(h_prev, self.Wh) + self.b)
        return h_next

class ManualRNNLayer:
    def __init__(self, units, Wx, Wh, b, activation=tanh, return_sequences=False, go_backwards=False):
        self.cell = ManualRNNCell(Wx, Wh, b, activation)
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        self.units = units

    def forward(self, X_embedded_batch):
        N, seq_len, input_dim = X_embedded_batch.shape
        h_t = np.zeros((N, self.units))
        outputs = []

        time_steps = range(seq_len)
        if self.go_backwards:
            time_steps = reversed(time_steps)

        for t in time_steps:
            xt = X_embedded_batch[:, t, :]
            h_t = self.cell.forward(xt, h_t)
            if self.return_sequences:
                outputs.append(h_t)

        if self.return_sequences:
            stacked_outputs = np.stack(outputs, axis=1)
            return stacked_outputs[:, ::-1, :] if self.go_backwards else stacked_outputs
        else:
            return h_t

class ManualBidirectionalRNNWrapper:
    def __init__(self, forward_rnn_layer, backward_rnn_layer, merge_mode='concat'):
        self.forward_rnn_layer = forward_rnn_layer
        self.backward_rnn_layer = backward_rnn_layer
        if merge_mode != 'concat':
            raise NotImplementedError("Only 'concat' merge mode is implemented for manual Bidirectional RNN.")
        self.merge_mode = merge_mode

    def forward(self, X_embedded_batch):
        # Forward
        output_forward = self.forward_rnn_layer.forward(X_embedded_batch)
        # Backward: reverse sequence on time axis for input, then reverse output back
        X_reversed = X_embedded_batch[:, ::-1, :]
        output_backward = self.backward_rnn_layer.forward(X_reversed)
        # If return_sequences, output shape (N, T, units), else (N, units)
        if self.forward_rnn_layer.return_sequences:
            # Reverse backward output on time axis to match forward
            output_backward = output_backward[:, ::-1, :]
        return np.concatenate([output_forward, output_backward], axis=-1)

class SimpleRNNFromScratch:
    def __init__(self):
        self.layers = []
        self.is_built = False
        self.vectorizer = None

    def build_from_keras(self, keras_model, vectorizer=None):
        # Allow vectorizer to be None for inference-only use
        self.vectorizer = vectorizer
        if vectorizer is not None:
            self.vocabulary_size = vectorizer.vocabulary_size()
        else:
            self.vocabulary_size = None

        for layer in keras_model.layers:
            layer_name = layer.__class__.__name__

            if layer_name == 'Embedding':
                embedding_weights = layer.get_weights()[0]
                self.layers.append(ManualEmbeddingLayer(embedding_weights))

            elif layer_name == 'SimpleRNN':
                weights = layer.get_weights()
                Wx, Wh, b = weights[0], weights[1], weights[2]
                activation_fn = tanh if layer.activation.__name__ == 'tanh' else sigmoid
                return_sequences = layer.return_sequences

                self.layers.append(ManualRNNLayer(
                    units=layer.units,
                    Wx=Wx,
                    Wh=Wh,
                    b=b,
                    activation=activation_fn,
                    return_sequences=return_sequences
                ))

            elif layer_name == 'Bidirectional':
                forward_layer = layer.forward_layer
                backward_layer = layer.backward_layer

                fw_weights = forward_layer.get_weights()
                fw_Wx, fw_Wh, fw_b = fw_weights[0], fw_weights[1], fw_weights[2]
                fw_activation_fn = tanh if forward_layer.activation.__name__ == 'tanh' else sigmoid
                fw_return_sequences = forward_layer.return_sequences

                bw_weights = backward_layer.get_weights()
                bw_Wx, bw_Wh, bw_b = bw_weights[0], bw_weights[1], bw_weights[2]
                bw_activation_fn = tanh if backward_layer.activation.__name__ == 'tanh' else sigmoid
                bw_return_sequences = backward_layer.return_sequences

                self.layers.append(ManualBidirectionalRNNWrapper(
                    ManualRNNLayer(
                        units=forward_layer.units,
                        Wx=fw_Wx,
                        Wh=fw_Wh,
                        b=fw_b,
                        activation=fw_activation_fn,
                        return_sequences=fw_return_sequences
                    ),
                    ManualRNNLayer(
                        units=backward_layer.units,
                        Wx=bw_Wx,
                        Wh=bw_Wh,
                        b=bw_b,
                        activation=bw_activation_fn,
                        return_sequences=bw_return_sequences
                    )
                ))

            elif layer_name == 'Dense':
                weights, bias = layer.get_weights()
                activation_fn = None
                if layer.activation.__name__ == 'relu':
                    activation_fn = relu
                elif layer.activation.__name__ == 'softmax':
                    activation_fn = softmax

                self.layers.append(ManualDenseLayer(weights, bias, activation_fn))

        self.is_built = True

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def predict(self, x):
        if not self.is_built:
            raise ValueError("Model not built. Call build_from_keras first.")
        return self.forward(x)

MAX_FEATURES = 10000
MAX_LEN = 200
EMBEDDING_DIM = 128
NUSAX_DATA_DIR = 'dataset/'

def load_and_preprocess_nusax_sentiment(max_features=MAX_FEATURES, maxlen=MAX_LEN):
    print("Loading NusaX sentiment data for SimpleRNN...")
    try:
        train_df = pd.read_csv(os.path.join(NUSAX_DATA_DIR, "train.csv"))
        valid_df = pd.read_csv(os.path.join(NUSAX_DATA_DIR, "valid.csv"))
        test_df = pd.read_csv(os.path.join(NUSAX_DATA_DIR, "test.csv"))
    except FileNotFoundError:
        print(f"Error: One or more NusaX CSV files not found in {NUSAX_DATA_DIR}")
        raise
    x_train_text, y_train_raw = train_df['text'].astype(str).values, train_df['label'].values
    x_val_text, y_val_raw = valid_df['text'].astype(str).values, valid_df['label'].values
    x_test_text, y_test_raw = test_df['text'].astype(str).values, test_df['label'].values
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_val = label_encoder.transform(y_val_raw)
    y_test = label_encoder.transform(y_test_raw)
    num_classes = len(label_encoder.classes_)
    print(f"Labels encoded. Classes: {label_encoder.classes_} -> {list(range(num_classes))}")
    text_vectorizer = layers.TextVectorization(
        max_tokens=max_features,
        output_sequence_length=maxlen,
        name="text_vectorizer"
    )
    text_vectorizer.adapt(x_train_text)
    vocab_size = text_vectorizer.vocabulary_size()
    print(f"Vocabulary size: {vocab_size} (max_features was {max_features})")
    x_train = text_vectorizer(x_train_text)
    x_val = text_vectorizer(x_val_text)
    x_test = text_vectorizer(x_test_text)
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_val shape: {x_val.shape}, y_val shape: {y_val.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
    print(f"Number of classes: {num_classes}")
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), num_classes, vocab_size, text_vectorizer

def build_rnn_model(vocab_size, embedding_dim, max_len, num_classes, rnn_config, rnn_type='SimpleRNN', bidirectional=False):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, SimpleRNN, Bidirectional, Dense

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
    for units in rnn_config:
        if bidirectional:
            model.add(Bidirectional(SimpleRNN(units, return_sequences=False)))
        else:
            model.add(SimpleRNN(units, return_sequences=False))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def rnn_hyperparameter_analysis(x_train, y_train, x_val, y_val, x_test, y_test, num_classes, vocab_size, rnn_type_to_test='SimpleRNN'):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
    import os
    print(f"Performing hyperparameter analysis for {rnn_type_to_test}...")

    embedding_dim = 128
    rnn_units = 64

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=MAX_LEN),
        SimpleRNN(rnn_units, return_sequences=False),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=3, validation_data=(x_val, y_val), verbose=1)

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    model_save_path = f"best_keras_{rnn_type_to_test.lower()}.keras"
    model.save(model_save_path)
    print(f"Best Keras {rnn_type_to_test} model saved as {model_save_path}")

    return model

def build_simplernn_model(vocab_size, embedding_dim, maxlen, num_classes,
                        rnn_configs, dropout_rate=0.5):
    from tensorflow import keras
    from tensorflow.keras import layers
    model = keras.Sequential(name="Keras_SimpleRNN_Model")
    model.add(layers.Input(shape=(maxlen,), dtype="int32", name="input_tokens"))
    model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                              input_length=maxlen, name="embedding"))
    for i, config in enumerate(rnn_configs):
        units = config.get('units', 64)
        bidirectional = config.get('bidirectional', False)
        return_sequences = config.get('return_sequences', True if i < len(rnn_configs) - 1 else False)
        rnn_layer = layers.SimpleRNN(units, return_sequences=return_sequences, name=f"rnn_{i+1}")
        if bidirectional:
            model.add(layers.Bidirectional(rnn_layer, name=f"bi_rnn_{i+1}"))
        else:
            model.add(rnn_layer)
    model.add(layers.Dropout(dropout_rate, name="dropout"))
    model.add(layers.Dense(num_classes, activation='softmax', name="output_dense"))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_evaluate_simplernn_variant(
    x_train, y_train, x_val, y_val, x_test, y_test,
    num_classes, vocab_size, embedding_dim, max_len,
    rnn_config, epochs=5, batch_size=32, dropout_rate=0.2, description=""
):
    from tensorflow.keras import models, layers, callbacks
    model = models.Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=max_len))
    for i, layer_config in enumerate(rnn_config):
        units = layer_config.get('units', 64)
        bidirectional = layer_config.get('bidirectional', False)
        # Only last RNN layer should have return_sequences=False
        return_sequences = layer_config.get('return_sequences', True if i < len(rnn_config) - 1 else False)
        rnn_layer = layers.SimpleRNN(units, return_sequences=return_sequences)
        if bidirectional:
            rnn_layer = layers.Bidirectional(rnn_layer)
        model.add(rnn_layer)
    model.add(layers.Dropout(dropout_rate))  # Dropout lebih kecil agar tidak terlalu agresif
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-5, verbose=1)
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[early_stop, reduce_lr]
    )
    y_pred_proba = model.predict(x_test, verbose=0)
    f1_score = calculate_f1_macro(y_test, y_pred_proba, num_classes)
    print(f"Completed training for {description}. Macro F1-Score: {f1_score:.4f}")
    return model, f1_score, history