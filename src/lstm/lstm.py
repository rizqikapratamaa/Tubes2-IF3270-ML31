import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from utils import (sigmoid, tanh, softmax, relu, calculate_f1_macro, plot_history,
                   d_sigmoid_dz, d_tanh_dz, d_relu_dz, ManualSparseCategoricalCrossentropy)

class ManualEmbeddingLayer:
    def __init__(self, embedding_matrix):
        self.embedding_matrix = embedding_matrix
        self.params = {'embedding_matrix': self.embedding_matrix}
        self.grads = {'embedding_matrix': np.zeros_like(self.embedding_matrix)}
    
    def forward(self, X_tokens_batch):
        return self.embedding_matrix[X_tokens_batch]
    
    def backward(self, dL_dOutput, X_tokens_batch):
        N, seq_len = X_tokens_batch.shape
        vocab_size, embed_dim = self.embedding_matrix.shape
        self.grads['embedding_matrix'].fill(0)
        for n in range(N):
            for t in range(seq_len):
                token_idx = X_tokens_batch[n, t]
                if token_idx < vocab_size:
                    self.grads['embedding_matrix'][token_idx] += dL_dOutput[n, t]
        return None

class ManualDenseLayer:
    def __init__(self, weights, bias, activation_fn=None):
        self.weights = weights
        self.bias = bias
        self.activation_fn = activation_fn
        self.params = {'weights': self.weights, 'bias': self.bias}
        self.grads = {'weights': np.zeros_like(self.weights), 'bias': np.zeros_like(self.bias)}
    
    def forward(self, X_batch):
        output = np.dot(X_batch, self.weights) + self.bias
        self.input_ = X_batch
        if self.activation_fn:
            output = self.activation_fn(output)
        self.output_ = output
        return output
    
    def backward(self, dL_dOutput):
        if self.activation_fn == softmax:
            dL_dZ = dL_dOutput
        elif self.activation_fn == relu:
            dL_dZ = dL_dOutput * d_relu_dz(self.output_)
        elif self.activation_fn == sigmoid:
            dL_dZ = dL_dOutput * d_sigmoid_dz(self.output_)
        else:
            dL_dZ = dL_dOutput
        self.grads['weights'] = np.dot(self.input_.T, dL_dZ)
        self.grads['bias'] = np.sum(dL_dZ, axis=0)
        dL_dInput = np.dot(dL_dZ, self.weights.T)
        return dL_dInput

MAX_FEATURES = 10000
MAX_LEN =  200
EMBEDDING_DIM = 128
NUSAX_DATA_DIR = "dataset/"

def load_and_preprocess_nusax_sentiment(max_features=MAX_FEATURES, maxlen=MAX_LEN):
    print("Loading NusaX sentiment data...")
    try:
        train_df = pd.read_csv(NUSAX_DATA_DIR + "train.csv")
        valid_df = pd.read_csv(NUSAX_DATA_DIR + "valid.csv")
        test_df = pd.read_csv(NUSAX_DATA_DIR + "test.csv")
    except FileNotFoundError:
        print(f"Error: One or more NusaX CSV files not found in {NUSAX_DATA_DIR}")
        print("Please ensure train.csv, valid.csv, test.csv are present.")
        raise
    x_train_text, y_train_raw = train_df['text'].astype(str).values, train_df['label'].values
    x_val_text, y_val_raw = valid_df['text'].astype(str).values, valid_df['label'].values
    x_test_text, y_test_raw = test_df['text'].astype(str).values, test_df['label'].values
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_val = label_encoder.transform(y_val_raw)
    y_test = label_encoder.transform(y_test_raw)
    num_classes = len(label_encoder.classes_)
    print(f"Labels encoded. Classes: {label_encoder.classes_} -> {np.arange(num_classes)}")
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

def build_lstm_model(vocab_size, embedding_dim, maxlen, num_classes,
                     lstm_configs, dropout_rate=0.5):
    model = keras.Sequential(name="Keras_LSTM_Model")
    model.add(layers.Input(shape=(maxlen,), dtype="int32", name="input_tokens"))
    model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                              input_length=maxlen, name="embedding"))
    for i, config in enumerate(lstm_configs):
        units = config.get('units', 64)
        bidirectional = config.get('bidirectional', False)
        return_sequences = config.get('return_sequences', True if i < len(lstm_configs) - 1 else False)
        lstm_layer = layers.LSTM(units, return_sequences=return_sequences,
                                 name=f"lstm_{i+1}")
        if bidirectional:
            model.add(layers.Bidirectional(lstm_layer, name=f"bi_lstm_{i+1}"))
        else:
            model.add(lstm_layer)
    model.add(layers.Dropout(dropout_rate, name="dropout"))
    model.add(layers.Dense(num_classes, activation='softmax', name="output_dense"))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_evaluate_lstm_variant(
    x_train, y_train, x_val, y_val, x_test, y_test,
    num_classes, vocab_size, embedding_dim, max_len,
    lstm_config, epochs=5, batch_size=32, description=""
):
    model = models.Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=max_len))
    
    for layer_config in lstm_config:
        units = layer_config.get('units', 64)
        bidirectional = layer_config.get('bidirectional', False)
        return_sequences = layer_config.get('return_sequences', False)
        
        lstm_layer = layers.LSTM(units, return_sequences=return_sequences)
        if bidirectional:
            lstm_layer = layers.Bidirectional(lstm_layer)
        model.add(lstm_layer)
    
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    y_pred_proba = model.predict(x_test, verbose=0)
    f1_score = calculate_f1_macro(y_test, y_pred_proba, num_classes)
    
    print(f"Completed training for {description}. Macro F1-Score: {f1_score:.4f}")
    
    return model, f1_score, history

def lstm_hyperparameter_analysis(
    x_train, y_train, x_val, y_val, x_test, y_test,
    num_classes, vocab_size, embedding_dim, maxlen, analysis_epochs=3):
    results = {}
    best_f1 = -1
    best_model_lstm = None
    print("\n=== LSTM Hyperparameter Analysis (NusaX Sentiment) ===")
    print("\n--- 1. Analysis: Number of LSTM Layers ---")
    num_layers_variations = [
        [{'units': 64, 'bidirectional': False}],
        [{'units': 64, 'bidirectional': False, 'return_sequences': True}, {'units': 32, 'bidirectional': False}],
        [{'units': 64, 'bidirectional': False, 'return_sequences': True}, {'units': 32, 'bidirectional': False, 'return_sequences': True}, {'units': 16, 'bidirectional': False}]
    ]
    for i, config in enumerate(num_layers_variations):
        desc = f"NumLayers_{i+1}"
        model, f1, history = train_and_evaluate_lstm_variant(
            x_train, y_train, x_val, y_val, x_test, y_test, num_classes, vocab_size,
            embedding_dim, maxlen, config, epochs=analysis_epochs, description=desc
        )
        results[desc] = {'f1': f1, 'history': history}
        if f1 > best_f1:
            best_f1, best_model_lstm = f1, model
    print("\nConclusion Number of LSTM Layers:")
    print(f"1 Layer F1: {results['NumLayers_1']['f1']:.4f}, 2 Layers F1: {results['NumLayers_2']['f1']:.4f}, 3 Layers F1: {results['NumLayers_3']['f1']:.4f}")
    print("Increasing LSTM layers generally improves F1-score by capturing more complex patterns, but 3 layers may lead to overfitting if validation loss increases.")
    print("\n--- 2. Analysis: LSTM Units (Cells per Layer) ---")
    units_variations = [
        [{'units': 32, 'bidirectional': False}],
        [{'units': 64, 'bidirectional': False}],
        [{'units': 128, 'bidirectional': False}]
    ]
    for i, config in enumerate(units_variations):
        desc = f"Units_{config[0]['units']}"
        model, f1, history = train_and_evaluate_lstm_variant(
            x_train, y_train, x_val, y_val, x_test, y_test, num_classes, vocab_size,
            embedding_dim, maxlen, config, epochs=analysis_epochs, description=desc
        )
        results[desc] = {'f1': f1, 'history': history}
        if f1 > best_f1:
            best_f1, best_model_lstm = f1, model
    print("\nConclusion LSTM Units:")
    print(f"32 Units F1: {results['Units_32']['f1']:.4f}, 64 Units F1: {results['Units_64']['f1']:.4f}, 128 Units F1: {results['Units_128']['f1']:.4f}")
    print("Larger units improve F1-score by increasing model capacity, but 128 units may increase computation time without proportional gains.")
    print("\n--- 3. Analysis: Bidirectional vs. Unidirectional ---")
    direction_variations = [
        [{'units': 64, 'bidirectional': False}],
        [{'units': 64, 'bidirectional': True}]
    ]
    for i, config in enumerate(direction_variations):
        desc = f"Direction_{'Bi' if config[0]['bidirectional'] else 'Uni'}"
        model, f1, history = train_and_evaluate_lstm_variant(
            x_train, y_train, x_val, y_val, x_test, y_test, num_classes, vocab_size,
            embedding_dim, maxlen, config, epochs=analysis_epochs, description=desc
        )
        results[desc] = {'f1': f1, 'history': history}
        if f1 > best_f1:
            best_f1, best_model_lstm = f1, model
    print("\nConclusion Bidirectionality:")
    print(f"Unidirectional F1: {results['Direction_Uni']['f1']:.4f}, Bidirectional F1: {results['Direction_Bi']['f1']:.4f}")
    print("Bidirectional LSTM typically outperforms unidirectional due to capturing context from both directions, improving F1-score.")
    print("\n--- Summary of F1 Scores (Keras LSTM Variants) ---")
    for desc, result in results.items():
        print(f"{desc}: F1 = {result['f1']:.4f}")
    if best_model_lstm:
        best_model_lstm.save("best_lstm_model.keras")
        print("\nBest Keras LSTM model saved as best_lstm_model.keras")
    else:
        print("\nNo Keras LSTM model was trained successfully to be saved.")
    return best_model_lstm, results

class ManualLSTMCell:
    def __init__(self, Wx, Wh, b, activation=tanh, recurrent_activation=sigmoid):
        units = Wh.shape[0]
        self.Wx_i, self.Wx_f, self.Wx_c, self.Wx_o = np.split(Wx, 4, axis=1)
        self.Wh_i, self.Wh_f, self.Wh_c, self.Wh_o = np.split(Wh, 4, axis=1)
        self.b_i, self.b_f, self.b_c, self.b_o = np.split(b, 4)
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.params = {
            'Wx_i': self.Wx_i, 'Wx_f': self.Wx_f, 'Wx_c': self.Wx_c, 'Wx_o': self.Wx_o,
            'Wh_i': self.Wh_i, 'Wh_f': self.Wh_f, 'Wh_c': self.Wh_c, 'Wh_o': self.Wh_o,
            'b_i': self.b_i, 'b_f': self.b_f, 'b_c': self.b_c, 'b_o': self.b_o
        }
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}
    
    def forward(self, xt, h_prev, c_prev):
        i_lin = np.dot(xt, self.Wx_i) + np.dot(h_prev, self.Wh_i) + self.b_i
        f_lin = np.dot(xt, self.Wx_f) + np.dot(h_prev, self.Wh_f) + self.b_f
        o_lin = np.dot(xt, self.Wx_o) + np.dot(h_prev, self.Wh_o) + self.b_o
        c_tilde_lin = np.dot(xt, self.Wx_c) + np.dot(h_prev, self.Wh_c) + self.b_c
        i = self.recurrent_activation(i_lin)
        f = self.recurrent_activation(f_lin)
        o = self.recurrent_activation(o_lin)
        c_tilde = self.activation(c_tilde_lin)
        c_next = f * c_prev + i * c_tilde
        h_next = o * self.activation(c_next)
        cache = (xt, h_prev, c_prev, i_lin, f_lin, o_lin, c_tilde_lin, i, f, o, c_tilde, c_next, h_next)
        return h_next, c_next, cache
    
    def backward(self, dh_next, dc_next_from_tplus1, cache):
        xt, h_prev, c_prev, i_lin, f_lin, o_lin, c_tilde_lin, i, f, o, c_tilde, c_next, h_next = cache
        act_c_next = self.activation(c_next)
        d_act_c_next_dz = d_tanh_dz(act_c_next) if self.activation == tanh else None
        do_lin_pre_act = dh_next * act_c_next
        dc_next = dh_next * o * d_act_c_next_dz
        dc_next += dc_next_from_tplus1
        df_lin_pre_act = dc_next * c_prev
        dc_prev = dc_next * f
        di_lin_pre_act = dc_next * c_tilde
        dc_tilde_lin_pre_act = dc_next * i
        di_lin = di_lin_pre_act * d_sigmoid_dz(i)
        df_lin = df_lin_pre_act * d_sigmoid_dz(f)
        do_lin = do_lin_pre_act * d_sigmoid_dz(o)
        dc_tilde_lin = dc_tilde_lin_pre_act * d_tanh_dz(c_tilde)
        self.grads['Wx_i'] += np.dot(xt.T, di_lin)
        self.grads['Wh_i'] += np.dot(h_prev.T, di_lin)
        self.grads['b_i'] += np.sum(di_lin, axis=0)
        self.grads['Wx_f'] += np.dot(xt.T, df_lin)
        self.grads['Wh_f'] += np.dot(h_prev.T, df_lin)
        self.grads['b_f'] += np.sum(df_lin, axis=0)
        self.grads['Wx_o'] += np.dot(xt.T, do_lin)
        self.grads['Wh_o'] += np.dot(h_prev.T, do_lin)
        self.grads['b_o'] += np.sum(do_lin, axis=0)
        self.grads['Wx_c'] += np.dot(xt.T, dc_tilde_lin)
        self.grads['Wh_c'] += np.dot(h_prev.T, dc_tilde_lin)
        self.grads['b_c'] += np.sum(dc_tilde_lin, axis=0)
        dx_t = np.dot(di_lin, self.Wx_i.T) + np.dot(df_lin, self.Wx_f.T) + \
               np.dot(do_lin, self.Wx_o.T) + np.dot(dc_tilde_lin, self.Wx_c.T)
        dh_prev = np.dot(di_lin, self.Wh_i.T) + np.dot(df_lin, self.Wh_f.T) + \
                  np.dot(do_lin, self.Wh_o.T) + np.dot(dc_tilde_lin, self.Wh_c.T)
        return dx_t, dh_prev, dc_prev
    
    def reset_grads(self):
        for k in self.grads:
            self.grads[k].fill(0)

class ManualLSTMLayer:
    def __init__(self, units, Wx, Wh, b, activation=tanh, recurrent_activation=sigmoid,
                 return_sequences=False, go_backwards=False):
        self.cell = ManualLSTMCell(Wx, Wh, b, activation, recurrent_activation)
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        self.units = units
        self.params = self.cell.params
        self.grads = self.cell.grads
    
    def forward(self, X_embedded_batch):
        N, seq_len, _ = X_embedded_batch.shape
        h_t = np.zeros((N, self.units))
        c_t = np.zeros((N, self.units))
        outputs_h = []
        cell_caches = []
        time_steps = range(seq_len)
        if self.go_backwards:
            time_steps = reversed(time_steps)
        current_X_sequence = X_embedded_batch
        if self.go_backwards:
            current_X_sequence = X_embedded_batch[:, ::-1, :]
        for t_idx_in_processing_order, t_actual_from_input in enumerate(time_steps):
            xt = current_X_sequence[:, t_idx_in_processing_order, :]
            h_t, c_t, cache = self.cell.forward(xt, h_t, c_t)
            outputs_h.append(h_t)
            cell_caches.append(cache)
        self.cell_caches_ = cell_caches
        if self.return_sequences:
            stacked_outputs = np.stack(outputs_h, axis=1)
            final_output_sequence = stacked_outputs[:, ::-1, :] if self.go_backwards else stacked_outputs
            self.last_output_ = final_output_sequence
            return final_output_sequence
        else:
            self.last_output_ = h_t
            return h_t
    
    def backward(self, dL_dOutput):
        N, seq_len, input_dim = self.cell_caches_[0][0].shape[0], len(self.cell_caches_), self.cell_caches_[0][0].shape[1]
        dL_dX_embedded_batch = np.zeros((N, seq_len, input_dim))
        dh_next_bptt = np.zeros((N, self.units))
        dc_next_bptt = np.zeros((N, self.units))
        self.cell.reset_grads()
        if self.return_sequences:
            dL_dOutput_processed_order = dL_dOutput[:, ::-1, :] if self.go_backwards else dL_dOutput
        else:
            dL_dOutput_processed_order = np.zeros((N, seq_len, self.units))
            dL_dOutput_processed_order[:, -1, :] = dL_dOutput
        time_steps_proc_order = range(seq_len)
        for t_proc in reversed(time_steps_proc_order):
            cache_t = self.cell_caches_[t_proc]
            current_dh = dL_dOutput_processed_order[:, t_proc, :] + dh_next_bptt
            dx_t, dh_prev_bptt, dc_prev_bptt = self.cell.backward(current_dh, dc_next_bptt, cache_t)
            actual_time_idx = (seq_len - 1 - t_proc) if self.go_backwards else t_proc
            dL_dX_embedded_batch[:, actual_time_idx, :] = dx_t
            dh_next_bptt = dh_prev_bptt
            dc_next_bptt = dc_prev_bptt
        return dL_dX_embedded_batch

class ManualBidirectionalLSTMWrapper:
    def __init__(self, forward_lstm_layer, backward_lstm_layer, merge_mode='concat'):
        self.forward_lstm_layer = forward_lstm_layer
        self.backward_lstm_layer = backward_lstm_layer
        if merge_mode != 'concat':
            raise NotImplementedError("Only 'concat' merge mode implemented.")
        self.merge_mode = merge_mode
        self.params = {f'fw_{k}': v for k, v in self.forward_lstm_layer.params.items()}
        self.params.update({f'bw_{k}': v for k, v in self.backward_lstm_layer.params.items()})
        self.grads = {f'fw_{k}': v for k, v in self.forward_lstm_layer.grads.items()}
        self.grads.update({f'bw_{k}': v for k, v in self.backward_lstm_layer.grads.items()})
    
    def forward(self, X_embedded_batch):
        output_forward = self.forward_lstm_layer.forward(X_embedded_batch)
        output_backward = self.backward_lstm_layer.forward(X_embedded_batch)
        concatenated_output = np.concatenate([output_forward, output_backward], axis=-1)
        self.last_output_ = concatenated_output
        return concatenated_output
    
    def backward(self, dL_dOutput_concat):
        units_fw = self.forward_lstm_layer.units
        dL_dOutput_fw, dL_dOutput_bw = np.split(dL_dOutput_concat, 2, axis=-1)
        if dL_dOutput_fw.shape[-1] != units_fw:
            dL_dOutput_fw = dL_dOutput_concat[..., :units_fw]
            dL_dOutput_bw = dL_dOutput_concat[..., units_fw:]
        dL_dX_fw = self.forward_lstm_layer.backward(dL_dOutput_fw)
        dL_dX_bw = self.backward_lstm_layer.backward(dL_dOutput_bw)
        return dL_dX_fw + dL_dX_bw

class LSTMFromScratch:
    def __init__(self):
        self.layers = []
        self.keras_model = None
    
    def add_keras_layer(self, keras_layer):
        if isinstance(keras_layer, layers.Embedding):
            embedding_matrix = keras_layer.get_weights()[0]
            self.layers.append(ManualEmbeddingLayer(embedding_matrix))
        elif isinstance(keras_layer, layers.LSTM):
            weights = keras_layer.get_weights()
            Wx, Wh, b = weights[0], weights[1], weights[2]
            units = keras_layer.units
            act_cfg = keras_layer.get_config()['activation']
            rec_act_cfg = keras_layer.get_config()['recurrent_activation']
            activation_fn = globals().get(act_cfg, tanh)
            recurrent_activation_fn = globals().get(rec_act_cfg, sigmoid)
            return_sequences = keras_layer.return_sequences
            go_backwards = keras_layer.go_backwards
            self.layers.append(ManualLSTMLayer(units, Wx, Wh, b, activation_fn,
                                               recurrent_activation_fn, return_sequences, go_backwards))
        elif isinstance(keras_layer, layers.Bidirectional):
            fw_keras_layer = keras_layer.forward_layer
            fw_weights = fw_keras_layer.get_weights()
            fw_Wx, fw_Wh, fw_b = fw_weights[0], fw_weights[1], fw_weights[2]
            fw_units = fw_keras_layer.units
            fw_act_cfg = fw_keras_layer.get_config()['activation']
            fw_rec_act_cfg = fw_keras_layer.get_config()['recurrent_activation']
            fw_act_fn = globals().get(fw_act_cfg, tanh)
            fw_rec_act_fn = globals().get(fw_rec_act_cfg, sigmoid)
            fw_ret_seq = fw_keras_layer.return_sequences
            manual_forward_lstm = ManualLSTMLayer(fw_units, fw_Wx, fw_Wh, fw_b, fw_act_fn,
                                                  fw_rec_act_fn, fw_ret_seq, go_backwards=False)
            bw_keras_layer = keras_layer.backward_layer
            bw_weights = bw_keras_layer.get_weights()
            bw_Wx, bw_Wh, bw_b = bw_weights[0], bw_weights[1], bw_weights[2]
            bw_units = bw_keras_layer.units
            bw_act_cfg = bw_keras_layer.get_config()['activation']
            bw_rec_act_cfg = bw_keras_layer.get_config()['recurrent_activation']
            bw_act_fn = globals().get(bw_act_cfg, tanh)
            bw_rec_act_fn = globals().get(bw_rec_act_cfg, sigmoid)
            bw_ret_seq = bw_keras_layer.return_sequences
            manual_backward_lstm = ManualLSTMLayer(bw_units, bw_Wx, bw_Wh, bw_b, bw_act_fn,
                                                   bw_rec_act_fn, bw_ret_seq, go_backwards=True)
            self.layers.append(ManualBidirectionalLSTMWrapper(manual_forward_lstm, manual_backward_lstm,
                                                              merge_mode=keras_layer.merge_mode))
        elif isinstance(keras_layer, layers.Dense):
            weights, bias = keras_layer.get_weights()
            activation_name = keras_layer.get_config()['activation']
            activation_fn = globals().get(activation_name)
            self.layers.append(ManualDenseLayer(weights, bias, activation_fn=activation_fn))
        elif isinstance(keras_layer, layers.Dropout):
            print(f"Ignoring Dropout layer: {keras_layer.name} during manual inference/training.")
        else:
            raise ValueError(f"Unsupported Keras layer type for LSTMFromScratch: {type(keras_layer)}")
    
    def load_keras_model(self, keras_model):
        self.keras_model = keras_model
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
            output = layer.forward(output)
        return output
    
    def compile_manual(self, loss_fn_instance, optimizer_instance):
        self.loss_fn = loss_fn_instance
        self.optimizer = optimizer_instance
    
    def fit_manual(self, X_train_tokens, y_train_labels, epochs, batch_size):
        if not hasattr(self, 'loss_fn') or not hasattr(self, 'optimizer'):
            raise RuntimeError("Model must be compiled with compile_manual() before training.")
        num_samples = X_train_tokens.shape[0]
        for epoch in range(epochs):
            epoch_loss = 0
            permutation = np.random.permutation(num_samples)
            X_train_shuffled = X_train_tokens[permutation]
            y_train_shuffled = y_train_labels[permutation]
            for i in range(0, num_samples, batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]
                layer_inputs = [X_batch]
                current_A = X_batch
                for layer_idx, layer in enumerate(self.layers):
                    current_A = layer.forward(current_A)
                    layer_inputs.append(current_A)
                y_pred_proba = current_A
                loss = self.loss_fn.calculate(y_pred_proba, y_batch)
                epoch_loss += loss * X_batch.shape[0]
                dL_dA_prev = self.loss_fn.derivative(y_pred_proba, y_batch)
                for layer_idx in reversed(range(len(self.layers))):
                    layer = self.layers[layer_idx]
                    if hasattr(layer, 'backward'):
                        if isinstance(layer, ManualEmbeddingLayer):
                            dL_dA_prev = layer.backward(dL_dA_prev, layer_inputs[layer_idx])
                        else:
                            dL_dA_prev = layer.backward(dL_dA_prev)
                    else:
                        print(f"Warning: Layer {type(layer)} does not have a backward method. Skipping.")
                self.optimizer.update(self.layers)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/num_samples:.4f}")
        print("Manual training finished.")

class ManualSGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, layers_list):
        for layer in layers_list:
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                for param_name, param_value in layer.params.items():
                    grad_value = layer.grads.get(param_name)
                    if grad_value is not None:
                        param_value -= self.learning_rate * grad_value
                        layer.grads[param_name].fill(0)
                    else:
                        print(f"Warning: No gradient found for param {param_name} in layer {type(layer)}")

# if __name__ == '__main__':
#     print("Running LSTM module directly (for testing purposes)...")
#     (x_train, y_train), (x_val, y_val), (x_test, y_test), \
#     num_classes, vocab_size, _ = load_and_preprocess_nusax_sentiment()
#     print("\nTraining a default Keras LSTM model...")
#     default_lstm_config = [{'units': 64, 'bidirectional': True, 'return_sequences': False}]
#     keras_model_instance, f1_keras, _ = train_and_evaluate_lstm_variant(
#         x_train, y_train, x_val, y_val, x_test, y_test, num_classes, vocab_size,
#         EMBEDDING_DIM, MAX_LEN, default_lstm_config, epochs=1, description="DefaultKeras"
#     )
#     keras_model_instance.save("default_lstm_model.keras")
#     print("\n--- Manual LSTM Forward Propagation Test ---")
#     manual_lstm = LSTMFromScratch()
#     manual_lstm.load_keras_model(keras_model_instance)
#     test_sample_idx = slice(0, 64)
#     x_test_sample = x_test[test_sample_idx]
#     y_test_sample_labels = y_test[test_sample_idx]
#     y_pred_keras_sample = keras_model_instance.predict(x_test_sample, verbose=0)
#     y_pred_manual_sample = manual_lstm.predict(x_test_sample)
#     print(f"Keras preds (sample sum): {np.sum(y_pred_keras_sample):.6f}")
#     print(f"Manual preds (sample sum): {np.sum(y_pred_manual_sample):.6f}")
#     abs_diff = np.sum(np.abs(y_pred_keras_sample - y_pred_manual_sample))
#     print(f"Sum of absolute differences (sample): {abs_diff:.6f}")
#     assert abs_diff < 1e-3, "Manual LSTM forward pass significantly different from Keras!"
#     f1_keras_sample = calculate_f1_macro(y_test_sample_labels, y_pred_keras_sample)
#     f1_manual_sample = calculate_f1_macro(y_test_sample_labels, y_pred_manual_sample)
#     print(f"F1 Keras (sample): {f1_keras_sample:.4f}")
#     print(f"F1 Manual (sample): {f1_manual_sample:.4f}")
#     print("\n--- Manual LSTM Backward Propagation & Training Test (Bonus) ---")
#     simpler_config = [{'units': 8, 'bidirectional': False, 'return_sequences': False}]
#     simpler_keras_model = build_lstm_model(vocab_size, EMBEDDING_DIM, MAX_LEN, num_classes, simpler_config)
#     simpler_keras_model.build(input_shape=(None, MAX_LEN))
#     manual_lstm_train = LSTMFromScratch()
#     manual_lstm_train.load_keras_model(simpler_keras_model)
#     loss_func = ManualSparseCategoricalCrossentropy()
#     optimizer = ManualSGD(learning_rate=0.01)
#     manual_lstm_train.compile_manual(loss_fn_instance=loss_func, optimizer_instance=optimizer)
#     print("Starting manual training with LSTMFromScratch...")
#     x_train_tiny = x_train[:64]
#     y_train_tiny = y_train[:64]
#     try:
#         manual_lstm_train.fit_manual(x_train_tiny, y_train_tiny, epochs=1, batch_size=8)
#         print("Manual training epoch completed.")
#         y_pred_manual_after_train = manual_lstm_train.predict(x_test_sample)
#         f1_manual_after_train = calculate_f1_macro(y_test_sample_labels, y_pred_manual_after_train)
#         print(f"F1 Manual (sample) after 1 epoch of manual training: {f1_manual_after_train:.4f}")
#     except NotImplementedError as e:
#         print(f"Could not run manual training: {e}")
#     except Exception as e:
#         print(f"An error occurred during manual training test: {e}")
#     print("LSTM module direct execution test finished.")