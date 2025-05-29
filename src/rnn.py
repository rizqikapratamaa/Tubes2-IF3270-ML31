import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils import sigmoid, tanh, softmax, relu

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
        output_forward = self.forward_rnn_layer.forward(X_embedded_batch)
        output_backward = self.backward_rnn_layer.forward(X_embedded_batch)
        return np.concatenate([output_forward, output_backward], axis=-1)

class SimpleRNNFromScratch:
    def __init__(self):
        self.layers = []
        self.is_built = False
        self.vectorizer = None

    def build_from_keras(self, keras_model, vectorizer):
        self.vectorizer = vectorizer
        self.vocabulary_size = vectorizer.vocabulary_size()

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

def load_and_preprocess_imdb(num_words=10000, maxlen=200):
    from tensorflow.keras.datasets import imdb
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_test = pad_sequences(x_test, maxlen=maxlen)

    val_size = int(len(x_test) * 0.5)
    x_val, y_val = x_test[:val_size], y_test[:val_size]
    x_test, y_test = x_test[val_size:], y_test[val_size:]

    num_classes = len(set(y_train))
    vocab_size = num_words

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), num_classes, vocab_size

MAX_FEATURES = 10000
MAX_LEN = 200

def rnn_hyperparameter_analysis(x_train, y_train, x_val, y_val, x_test, y_test, num_classes, vocab_size, rnn_type_to_test='SimpleRNN'):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

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

    return model