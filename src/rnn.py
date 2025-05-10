import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils import sigmoid, tanh, softmax, calculate_f1_macro, plot_history
from sklearn.model_selection import train_test_split

class ManualEmbeddingLayer:
    def __init__(self, embedding_matrix):
        self.embedding_matrix = embedding_matrix # Shape: (vocab_size, embedding_dim)

    def forward(self, X_tokens_batch): # X_tokens_batch shape: (N, seq_len)
        return self.embedding_matrix[X_tokens_batch] # Output shape: (N, seq_len, embedding_dim)

class ManualSimpleRNNCell:
    def __init__(self, Wx, Wh, b, activation=tanh):
        self.Wx = Wx  # Kernel for input, shape (input_dim, units)
        self.Wh = Wh  # Kernel for hidden state, shape (units, units)
        self.b = b    # Bias, shape (units,)
        self.activation = activation

    def forward(self, xt, h_prev): # xt: (N, input_dim), h_prev: (N, units)
        # print(f"xt: {xt.shape}, Wx: {self.Wx.shape}, h_prev: {h_prev.shape}, Wh: {self.Wh.shape}, b: {self.b.shape}")
        h_next = self.activation(np.dot(xt, self.Wx) + np.dot(h_prev, self.Wh) + self.b)
        return h_next, h_next

class ManualSimpleRNNLayer:
    def __init__(self, units, Wx, Wh, b, activation=tanh, return_sequences=False, go_backwards=False):
        self.cell = ManualSimpleRNNCell(Wx, Wh, b, activation)
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        self.units = units

    def forward(self, X_embedded_batch): # X_embedded_batch: (N, seq_len, input_dim)
        N, seq_len, input_dim = X_embedded_batch.shape
        
        h_t = np.zeros((N, self.units))
        
        outputs = []

        time_steps = range(seq_len)
        if self.go_backwards:
            time_steps = reversed(time_steps)

        for t in time_steps:
            xt = X_embedded_batch[:, t, :] # Input at time t for all batches: (N, input_dim)
            h_t, _ = self.cell.forward(xt, h_t)
            if self.return_sequences:
                outputs.append(h_t)
        
        if self.return_sequences:
            stacked_outputs = np.stack(outputs, axis=1) 
            return stacked_outputs[:, ::-1, :] if self.go_backwards else stacked_outputs
        else:
            return h_t # Last hidden state (N, units)

class ManualBidirectionalRNNWrapper:
    def __init__(self, forward_rnn_layer, backward_rnn_layer, merge_mode='concat'):
        self.forward_rnn_layer = forward_rnn_layer
        self.backward_rnn_layer = backward_rnn_layer
        if merge_mode != 'concat':
            raise NotImplementedError("Only 'concat' merge mode is implemented for manual Bidirectional RNN.")
        self.merge_mode = merge_mode

    def forward(self, X_embedded_batch):
        # print(f"Bidirectional input shape: {X_embedded_batch.shape}")
        output_forward = self.forward_rnn_layer.forward(X_embedded_batch)
        output_backward = self.backward_rnn_layer.forward(X_embedded_batch)
        
        # print(f"Forward output: {output_forward.shape}, Backward output: {output_backward.shape}")
        # If return_sequences=True, shapes are (N, seq_len, units). Concat to (N, seq_len, 2*units)
        # If return_sequences=False, shapes are (N, units). Concat to (N, 2*units)
        return np.concatenate([output_forward, output_backward], axis=-1)

class ManualDenseLayer:
    def __init__(self, weights, bias, activation_fn=None):
        self.weights = weights # Shape: (input_features, output_features)
        self.bias = bias # Shape: (output_features,)
        self.activation_fn = activation_fn

    def forward(self, X_batch): # X_batch shape: (N, input_features)
        output = np.dot(X_batch, self.weights) + self.bias
        if self.activation_fn:
            output = self.activation_fn(output)
        return output

class SimpleRNNFromScratch:
    def __init__(self):
        self.layers = []

    def add_keras_layer(self, keras_layer):
        if isinstance(keras_layer, layers.Embedding):
            embedding_matrix = keras_layer.get_weights()[0]
            self.layers.append(ManualEmbeddingLayer(embedding_matrix))
        
        elif isinstance(keras_layer, layers.SimpleRNN):
            weights = keras_layer.get_weights()
            Wx, Wh, b = weights[0], weights[1], weights[2]
            units = keras_layer.units
            activation_name = keras_layer.activation.__name__
            activation_fn = tanh if activation_name == 'tanh' else sigmoid
            return_sequences = keras_layer.return_sequences
            self.layers.append(ManualSimpleRNNLayer(units, Wx, Wh, b, activation_fn, return_sequences))

        elif isinstance(keras_layer, layers.Bidirectional):
            forward_keras_layer = keras_layer.forward_layer
            backward_keras_layer = keras_layer.backward_layer

            fw_weights = forward_keras_layer.get_weights()
            fw_Wx, fw_Wh, fw_b = fw_weights[0], fw_weights[1], fw_weights[2]
            fw_units = forward_keras_layer.units
            fw_activation_name = forward_keras_layer.activation.__name__
            fw_activation_fn = tanh if fw_activation_name == 'tanh' else sigmoid
            fw_return_sequences = forward_keras_layer.return_sequences
            
            manual_forward_rnn = ManualSimpleRNNLayer(fw_units, fw_Wx, fw_Wh, fw_b, 
                                                    fw_activation_fn, fw_return_sequences, go_backwards=False)

            bw_weights = backward_keras_layer.get_weights() 
            bw_Wx, bw_Wh, bw_b = bw_weights[0], bw_weights[1], bw_weights[2]
            bw_units = backward_keras_layer.units
            bw_activation_name = backward_keras_layer.activation.__name__
            bw_activation_fn = tanh if bw_activation_name == 'tanh' else sigmoid
            bw_return_sequences = backward_keras_layer.return_sequences

            manual_backward_rnn = ManualSimpleRNNLayer(bw_units, bw_Wx, bw_Wh, bw_b, 
                                                     bw_activation_fn, bw_return_sequences, go_backwards=True)
            
            self.layers.append(ManualBidirectionalRNNWrapper(manual_forward_rnn, manual_backward_rnn))

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
        
        elif isinstance(keras_layer, (layers.Flatten, layers.GlobalAveragePooling1D)):
             print(f"Ignoring layer not typically used in this RNN setup or needs specific handling: {type(keras_layer)}")

        else:
            raise ValueError(f"Unsupported Keras layer type for SimpleRNNFromScratch: {type(keras_layer)}")

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
            # print(f"Applying layer: {type(layer)}, input shape: {output.shape if isinstance(output, np.ndarray) else 'N/A'}")
            output = layer.forward(output)
            # print(f"Output shape: {output.shape}")
        return output

MAX_FEATURES = 10000  # Max vocabulary size
MAX_LEN = 200         # Max sequence length for padding/truncating
EMBEDDING_DIM = 128

def load_and_preprocess_imdb(num_words=MAX_FEATURES, maxlen=MAX_LEN):
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=num_words)
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
    
    # For NusaX, use TextVectorization on your text data first
    # Example for NusaX (conceptual):
    # df = pd.read_csv('path_to_nusax_train.csv') 
    # texts = df['text'].values
    # labels = df['label'].values 
    # vectorizer = layers.TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=MAX_LEN)
    # vectorizer.adapt(texts)
    # x_train_vec = vectorizer(texts)
    # ... then split into train/val ...
    
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    num_classes = len(np.unique(y_train)) # For IMDB, this is 2 (positive/negative)
                                        # For NusaX, it could be 3 (positive/negative/neutral)
    
    print(f"x_train shape: {x_train.shape}")
    print(f"x_val shape: {x_val.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"Number of classes: {num_classes}")
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), num_classes, MAX_FEATURES


def build_rnn_model(vocab_size, embedding_dim, max_len, num_classes,
                    rnn_layers_config, rnn_type='SimpleRNN', bidirectional=False):
    model = keras.Sequential()
    model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len, name="embedding"))

    RNNLayer = layers.SimpleRNN if rnn_type == 'SimpleRNN' else layers.LSTM

    for i, units in enumerate(rnn_layers_config):
        is_last_rnn_layer = (i == len(rnn_layers_config) - 1)
        return_sequences = not is_last_rnn_layer 
        
        rnn_layer = RNNLayer(units, return_sequences=return_sequences, name=f'{rnn_type.lower()}{i+1}')
        
        if bidirectional:
            model.add(layers.Bidirectional(rnn_layer, name=f'bi_{rnn_type.lower()}{i+1}'))
        else:
            model.add(rnn_layer)
        model.add(layers.Dropout(0.3, name=f'dropout_{i+1}')) 

    # Output layer
    # For binary classification (IMDB), a single sigmoid unit is common
    # For multi-class (like NusaX if it has 3+ classes), use softmax
    if num_classes == 2:
        model.add(layers.Dense(1, activation='sigmoid', name='output_dense'))
        loss_fn = 'binary_crossentropy'
    else: # Multiclass
        model.add(layers.Dense(num_classes, activation='softmax', name='output_dense'))
        loss_fn = 'sparse_categorical_crossentropy'
        
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    return model

def train_and_evaluate_rnn_variant(
    x_train, y_train, x_val, y_val, x_test, y_test, num_classes, vocab_size,
    rnn_layers_config, rnn_type='SimpleRNN', bidirectional=False, 
    epochs=5, batch_size=64, description=""):

    print(f"\n--- Training RNN: {description} ---")
    print(f"RNN Layers: {rnn_layers_config}, Type: {rnn_type}, Bidirectional: {bidirectional}")

    model = build_rnn_model(vocab_size, EMBEDDING_DIM, MAX_LEN, num_classes,
                            rnn_layers_config, rnn_type, bidirectional)
    model.summary()
    
    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_val, y_val),
                        verbose=1)

    plot_history(history, title_prefix=f"RNN {description}")

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    y_pred_keras_proba = model.predict(x_test)
    
    # Adjust for binary vs multiclass f1 score calculation
    if num_classes == 2: # Binary
        y_pred_keras_classes = (y_pred_keras_proba > 0.5).astype(int)
        f1_keras = calculate_f1_macro(y_test, y_pred_keras_classes, num_classes)
    else: # Multiclass
        f1_keras = calculate_f1_macro(y_test, y_pred_keras_proba, num_classes)

    print(f"Test Accuracy (Keras): {test_acc:.4f}")
    print(f"Macro F1-Score (Keras): {f1_keras:.4f}")
    
    return model, f1_keras, history


def rnn_hyperparameter_analysis(x_train, y_train, x_val, y_val, x_test, y_test, num_classes, vocab_size, rnn_type_to_test='SimpleRNN'): 
    results_rnn = {}
    best_f1_rnn = -1
    best_model_rnn = None

    print(f"\n=== 1. Analisis Pengaruh Jumlah Layer {rnn_type_to_test} ===") 
    num_rnn_layer_variations = [
        [64],            
        [64, 32],        
        [64, 32, 32]     
    ]
    for i, config in enumerate(num_rnn_layer_variations):
        desc = f"Num{rnn_type_to_test}Layers_{i+1}" 
        model, f1, _ = train_and_evaluate_rnn_variant(
            x_train, y_train, x_val, y_val, x_test, y_test, num_classes, vocab_size,
            rnn_layers_config=config, rnn_type=rnn_type_to_test, bidirectional=False, 
            epochs=3, description=desc 
        )
        results_rnn[desc] = f1
        if f1 > best_f1_rnn: best_f1_rnn = f1; best_model_rnn = model
    print(f"Kesimpulan Jumlah Layer {rnn_type_to_test}: [Analisis Anda di sini]") 

    print(f"\n=== 2. Analisis Pengaruh Banyak Cell {rnn_type_to_test} ===") 
    cell_variations = [
        [32],         
        [64],         
        [128]         
    ]
    for i, config in enumerate(cell_variations):
        desc = f"Num{rnn_type_to_test}Cells_{i+1}" 
        model, f1, _ = train_and_evaluate_rnn_variant(
            x_train, y_train, x_val, y_val, x_test, y_test, num_classes, vocab_size,
            rnn_layers_config=config, rnn_type=rnn_type_to_test, bidirectional=False, 
            epochs=3, description=desc
        )
        results_rnn[desc] = f1
        if f1 > best_f1_rnn: best_f1_rnn = f1; best_model_rnn = model
    print(f"Kesimpulan Banyak Cell {rnn_type_to_test}: [Analisis Anda di sini]") 

    print(f"\n=== 3. Analisis Pengaruh Arah Layer {rnn_type_to_test} ===") 
    direction_variations = [False, True] 
    base_rnn_config = [64] 
    for bi_enabled in direction_variations:
        direction_label = "Bi" if bi_enabled else "Uni"
        desc = f"{rnn_type_to_test}Direction_{direction_label}" 
        model, f1, _ = train_and_evaluate_rnn_variant(
            x_train, y_train, x_val, y_val, x_test, y_test, num_classes, vocab_size,
            rnn_layers_config=base_rnn_config, rnn_type=rnn_type_to_test, bidirectional=bi_enabled, 
            epochs=3, description=desc
        )
        results_rnn[desc] = f1
        if f1 > best_f1_rnn: best_f1_rnn = f1; best_model_rnn = model
    print(f"Kesimpulan Arah Layer {rnn_type_to_test}: [Analisis Anda di sini]") 

    print(f"\n--- Ringkasan F1 Scores {rnn_type_to_test} Variants ---") 
    for desc, f1_val in results_rnn.items():
        print(f"{desc}: {f1_val:.4f}")
        
    if best_model_rnn:
        best_model_rnn.save(f"best_{rnn_type_to_test.lower()}_model.keras") 
        print(f"\nBest Keras {rnn_type_to_test} model saved as best_{rnn_type_to_test.lower()}_model.keras") 
    else:
        print(f"\nNo {rnn_type_to_test} model was trained successfully.") 
    return best_model_rnn
