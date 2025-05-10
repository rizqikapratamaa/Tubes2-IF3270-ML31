import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils import sigmoid, tanh, softmax, relu, calculate_f1_macro, plot_history
from rnn import ManualEmbeddingLayer, ManualDenseLayer, load_and_preprocess_imdb # Re-use
# We can also re-use load_and_preprocess_imdb, build_rnn_model (with rnn_type='LSTM'),
# train_and_evaluate_rnn_variant (with rnn_type='LSTM'), and rnn_hyperparameter_analysis (with rnn_type='LSTM')
# from rnn_model import load_and_preprocess_imdb, build_rnn_model, train_and_evaluate_rnn_variant, rnn_hyperparameter_analysis
# For clarity, I will redefine the specific LSTM cell and layer here.
# If you were building a library, you'd factor out commonalities more.

class ManualLSTMCell:
    def __init__(self, Wx, Wh, b, activation=tanh, recurrent_activation=sigmoid):
        # Keras LSTM weights: kernel (Wx), recurrent_kernel (Wh), bias (b)
        # Each is a concatenation for i, f, c, o gates.
        # Wx shape: (input_dim, 4 * units)
        # Wh shape: (units, 4 * units)
        # b shape: (4 * units,)
        units = Wh.shape[0] # or b.shape[0] // 4

        self.Wx_i = Wx[:, :units]
        self.Wx_f = Wx[:, units:2*units]
        self.Wx_c = Wx[:, 2*units:3*units]
        self.Wx_o = Wx[:, 3*units:]

        self.Wh_i = Wh[:, :units]
        self.Wh_f = Wh[:, units:2*units]
        self.Wh_c = Wh[:, 2*units:3*units]
        self.Wh_o = Wh[:, 3*units:]

        # Bias can be split similarly or Keras might provide it split.
        # Keras usually provides a single bias vector.
        self.b_i = b[:units]
        self.b_f = b[units:2*units]
        self.b_c = b[2*units:3*units]
        self.b_o = b[3*units:]
        
        self.activation = activation                 # Typically tanh
        self.recurrent_activation = recurrent_activation # Typically sigmoid

    def forward(self, xt, h_prev, c_prev): # xt:(N,input_dim), h_prev:(N,units), c_prev:(N,units)
        # print(f"LSTMCell xt: {xt.shape}, h_prev: {h_prev.shape}, c_prev: {c_prev.shape}")
        # print(f"Wx_i: {self.Wx_i.shape}, Wh_i: {self.Wh_i.shape}, b_i: {self.b_i.shape}")

        i = self.recurrent_activation(np.dot(xt, self.Wx_i) + np.dot(h_prev, self.Wh_i) + self.b_i)
        f = self.recurrent_activation(np.dot(xt, self.Wx_f) + np.dot(h_prev, self.Wh_f) + self.b_f)
        o = self.recurrent_activation(np.dot(xt, self.Wx_o) + np.dot(h_prev, self.Wh_o) + self.b_o)
        c_tilde = self.activation(np.dot(xt, self.Wx_c) + np.dot(h_prev, self.Wh_c) + self.b_c)
        
        c_next = f * c_prev + i * c_tilde
        h_next = o * self.activation(c_next)
        
        return h_next, c_next

class ManualLSTMLayer:
    def __init__(self, units, Wx, Wh, b, activation=tanh, recurrent_activation=sigmoid,
                 return_sequences=False, go_backwards=False):
        self.cell = ManualLSTMCell(Wx, Wh, b, activation, recurrent_activation)
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        self.units = units

    def forward(self, X_embedded_batch): # X_embedded_batch: (N, seq_len, input_dim)
        N, seq_len, input_dim = X_embedded_batch.shape
        
        h_t = np.zeros((N, self.units)) # Initial hidden state
        c_t = np.zeros((N, self.units)) # Initial cell state
        
        outputs = []
        time_steps = range(seq_len)
        if self.go_backwards:
            time_steps = reversed(time_steps)

        for t in time_steps:
            xt = X_embedded_batch[:, t, :] # Input at time t for all batches: (N, input_dim)
            h_t, c_t = self.cell.forward(xt, h_t, c_t)
            if self.return_sequences:
                outputs.append(h_t)
        
        if self.return_sequences:
            stacked_outputs = np.stack(outputs, axis=1)
            return stacked_outputs[:, ::-1, :] if self.go_backwards else stacked_outputs
        else:
            return h_t # Last hidden state (N, units)

class ManualBidirectionalLSTMWrapper: # Similar to RNN one
    def __init__(self, forward_lstm_layer, backward_lstm_layer, merge_mode='concat'):
        self.forward_lstm_layer = forward_lstm_layer
        self.backward_lstm_layer = backward_lstm_layer
        if merge_mode != 'concat':
            raise NotImplementedError("Only 'concat' merge mode is implemented for manual Bidirectional LSTM.")
        self.merge_mode = merge_mode

    def forward(self, X_embedded_batch):
        output_forward = self.forward_lstm_layer.forward(X_embedded_batch)
        output_backward = self.backward_lstm_layer.forward(X_embedded_batch)
        return np.concatenate([output_forward, output_backward], axis=-1)

class LSTMFromScratch:
    def __init__(self):
        self.layers = []

    def add_keras_layer(self, keras_layer):
        if isinstance(keras_layer, layers.Embedding):
            embedding_matrix = keras_layer.get_weights()[0]
            self.layers.append(ManualEmbeddingLayer(embedding_matrix))
        
        elif isinstance(keras_layer, layers.LSTM):
            weights = keras_layer.get_weights()
            Wx, Wh, b = weights[0], weights[1], weights[2]
            units = keras_layer.units
            
            activation_name = keras_layer.activation.__name__
            activation_fn = tanh if activation_name == 'tanh' else None # Add more
            
            recurrent_activation_name = keras_layer.recurrent_activation.__name__
            recurrent_activation_fn = sigmoid if recurrent_activation_name == 'sigmoid' else None # Add more

            return_sequences = keras_layer.return_sequences
            self.layers.append(ManualLSTMLayer(units, Wx, Wh, b, activation_fn, 
                                               recurrent_activation_fn, return_sequences))

        elif isinstance(keras_layer, layers.Bidirectional):
            forward_keras_layer = keras_layer.forward_layer
            backward_keras_layer = keras_layer.backward_layer

            # Forward LSTM
            fw_weights = forward_keras_layer.get_weights()
            fw_Wx, fw_Wh, fw_b = fw_weights[0], fw_weights[1], fw_weights[2]
            fw_units = forward_keras_layer.units
            fw_act_name = forward_keras_layer.activation.__name__
            fw_act_fn = tanh if fw_act_name == 'tanh' else None
            fw_rec_act_name = forward_keras_layer.recurrent_activation.__name__
            fw_rec_act_fn = sigmoid if fw_rec_act_name == 'sigmoid' else None
            fw_ret_seq = forward_keras_layer.return_sequences
            manual_forward_lstm = ManualLSTMLayer(fw_units, fw_Wx, fw_Wh, fw_b, fw_act_fn,
                                                  fw_rec_act_fn, fw_ret_seq, go_backwards=False)

            # Backward LSTM
            bw_weights = backward_keras_layer.get_weights()
            bw_Wx, bw_Wh, bw_b = bw_weights[0], bw_weights[1], bw_weights[2]
            bw_units = backward_keras_layer.units
            bw_act_name = backward_keras_layer.activation.__name__
            bw_act_fn = tanh if bw_act_name == 'tanh' else None
            bw_rec_act_name = backward_keras_layer.recurrent_activation.__name__
            bw_rec_act_fn = sigmoid if bw_rec_act_name == 'sigmoid' else None
            bw_ret_seq = backward_keras_layer.return_sequences
            manual_backward_lstm = ManualLSTMLayer(bw_units, bw_Wx, bw_Wh, bw_b, bw_act_fn,
                                                   bw_rec_act_fn, bw_ret_seq, go_backwards=True)
            
            self.layers.append(ManualBidirectionalLSTMWrapper(manual_forward_lstm, manual_backward_lstm))

        elif isinstance(keras_layer, layers.Dense):
            weights, bias = keras_layer.get_weights()
            activation_fn = None
            activation_name = keras_layer.get_config()['activation']
            if activation_name == 'relu': activation_fn = relu
            elif activation_name == 'softmax': activation_fn = softmax
            elif activation_name == 'sigmoid': activation_fn = sigmoid
            self.layers.append(ManualDenseLayer(weights, bias, activation_fn=activation_fn))
        
        elif isinstance(keras_layer, layers.Dropout):
            print(f"Ignoring Dropout layer: {keras_layer.name} during manual inference")
        
        else:
            raise ValueError(f"Unsupported Keras layer type for LSTMFromScratch: {type(keras_layer)}")

    def load_keras_model(self, keras_model):
        self.layers = []
        for layer in keras_model.layers:
            print(f"Processing Keras layer: {layer.name} of type {type(layer)}")
            if isinstance(layer, tf.keras.layers.InputLayer):
                print(f"Skipping InputLayer: {layer.name}")
                continue
            self.add_keras_layer(layer)

    def predict(self, X_batch_tokens):
        output = X_batch_tokens
        for layer in self.layers:
            # print(f"Applying LSTM layer: {type(layer)}, input shape: {output.shape}")
            output = layer.forward(output)
            # print(f"Output shape: {output.shape}")
        return output

# Note: The Keras training and hyperparameter analysis for LSTM
# will use the functions from rnn_model.py by passing rnn_type='LSTM'.
# No need to redefine them here if they are imported.
# from rnn_model import load_and_preprocess_imdb, rnn_hyperparameter_analysis
# Define MAX_FEATURES, MAX_LEN, EMBEDDING_DIM if not imported
MAX_FEATURES = 10000
MAX_LEN = 200
EMBEDDING_DIM = 128
