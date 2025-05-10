import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils import relu, softmax, pad_with_zeros, calculate_f1_macro, plot_history
from sklearn.model_selection import train_test_split

class ManualConv2DLayer:
    def __init__(self, kernel, bias, stride=1, padding='valid'):
        self.kernel = kernel # Shape: (KH, KW, C_in, C_out)
        self.bias = bias # Shape: (C_out,)
        self.stride = stride
        self.padding = padding.lower()

    def forward(self, X_batch): # X_batch shape: (N, H_in, W_in, C_in)
        N, H_in, W_in, C_in = X_batch.shape
        KH, KW, _, C_out = self.kernel.shape

        if self.padding == 'same':
            out_height = int(np.ceil(float(H_in) / float(self.stride)))
            out_width  = int(np.ceil(float(W_in) / float(self.stride)))

            pad_h_total = max((out_height - 1) * self.stride + KH - H_in, 0)
            pad_w_total = max((out_width - 1) * self.stride + KW - W_in, 0)

            pad_top = pad_h_total // 2
            pad_bottom = pad_h_total - pad_top
            pad_left = pad_w_total // 2
            pad_right = pad_w_total - pad_left
            
            X_padded = np.pad(X_batch, ((0,0), (pad_top, pad_bottom), (pad_left, pad_right), (0,0)), mode='constant')
            H_in, W_in = X_padded.shape[1:3]
        elif self.padding == 'valid':
            X_padded = X_batch
            out_height = (H_in - KH) // self.stride + 1
            out_width = (W_in - KW) // self.stride + 1
        else:
            raise ValueError("Invalid padding mode")

        output = np.zeros((N, out_height, out_width, C_out))

        for n in range(N):
            for c_out_idx in range(C_out):
                for r_out in range(out_height):
                    for c_out in range(out_width):
                        r_start = r_out * self.stride
                        r_end = r_start + KH
                        c_start = c_out * self.stride
                        c_end = c_start + KW
                        
                        x_slice = X_padded[n, r_start:r_end, c_start:c_end, :]
                        
                        conv_sum = np.sum(x_slice * self.kernel[:, :, :, c_out_idx])
                        output[n, r_out, c_out, c_out_idx] = conv_sum
                output[n, :, :, c_out_idx] += self.bias[c_out_idx]
        return output

class ManualMaxPooling2DLayer:
    def __init__(self, pool_size=(2, 2), stride=None, padding='valid'): # Keras default
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size[0]
        self.padding = padding.lower()

    def forward(self, X_batch): # X_batch shape: (N, H_in, W_in, C_in)
        N, H_in, W_in, C_in = X_batch.shape
        PH, PW = self.pool_size

        if self.padding == 'same':
            out_height = int(np.ceil(float(H_in) / float(self.stride)))
            out_width  = int(np.ceil(float(W_in) / float(self.stride)))

            pad_h_total = max((out_height - 1) * self.stride + PH - H_in, 0)
            pad_w_total = max((out_width - 1) * self.stride + PW - W_in, 0)
            
            pad_top = pad_h_total // 2
            pad_bottom = pad_h_total - pad_top
            pad_left = pad_w_total // 2
            pad_right = pad_w_total - pad_left
            
            X_padded = np.pad(X_batch, ((0,0), (pad_top, pad_bottom), (pad_left, pad_right), (0,0)), mode='constant', constant_values=-np.inf)
            H_in, W_in = X_padded.shape[1:3]
        elif self.padding == 'valid':
            X_padded = X_batch
            out_height = (H_in - PH) // self.stride + 1
            out_width = (W_in - PW) // self.stride + 1
        else:
            raise ValueError("Invalid padding mode for MaxPooling")


        output = np.zeros((N, out_height, out_width, C_in))

        for n in range(N):
            for c_idx in range(C_in):
                for r_out in range(out_height):
                    for c_out in range(out_width):
                        r_start = r_out * self.stride
                        r_end = r_start + PH
                        c_start = c_out * self.stride
                        c_end = c_start + PW
                        
                        x_slice = X_padded[n, r_start:r_end, c_start:c_end, c_idx]
                        output[n, r_out, c_out, c_idx] = np.max(x_slice)
        return output

class ManualAveragePooling2DLayer:
    def __init__(self, pool_size=(2, 2), stride=None, padding='valid'):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size[0]
        self.padding = padding.lower()

    def forward(self, X_batch): # X_batch shape: (N, H_in, W_in, C_in)
        N, H_in, W_in, C_in = X_batch.shape
        PH, PW = self.pool_size

        if self.padding == 'same':
            out_height = int(np.ceil(float(H_in) / float(self.stride)))
            out_width  = int(np.ceil(float(W_in) / float(self.stride)))

            pad_h_total = max((out_height - 1) * self.stride + PH - H_in, 0)
            pad_w_total = max((out_width - 1) * self.stride + PW - W_in, 0)
            
            pad_top = pad_h_total // 2
            pad_bottom = pad_h_total - pad_top
            pad_left = pad_w_total // 2
            pad_right = pad_w_total - pad_left
            
            X_padded = np.pad(X_batch, ((0,0), (pad_top, pad_bottom), (pad_left, pad_right), (0,0)), mode='constant')
            H_in, W_in = X_padded.shape[1:3]
        elif self.padding == 'valid':
            X_padded = X_batch
            out_height = (H_in - PH) // self.stride + 1
            out_width = (W_in - PW) // self.stride + 1
        else:
            raise ValueError("Invalid padding mode for AvgPooling")

        output = np.zeros((N, out_height, out_width, C_in))

        for n in range(N):
            for c_idx in range(C_in):
                for r_out in range(out_height):
                    for c_out in range(out_width):
                        r_start = r_out * self.stride
                        r_end = r_start + PH
                        c_start = c_out * self.stride
                        c_end = c_start + PW
                        
                        x_slice = X_padded[n, r_start:r_end, c_start:c_end, c_idx]
                        output[n, r_out, c_out, c_idx] = np.mean(x_slice)
        return output

class ManualFlattenLayer:
    def forward(self, X_batch): # X_batch shape: (N, H, W, C)
        N = X_batch.shape[0]
        return X_batch.reshape(N, -1) # Reshape to (N, H*W*C)

class ManualGlobalAveragePooling2DLayer:
    def forward(self, X_batch): # X_batch shape: (N, H, W, C)
        return np.mean(X_batch, axis=(1, 2)) # Result shape (N, C)

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

class CNNFromScratch:
    def __init__(self):
        self.layers = []
        self.keras_layer_map = {
            layers.Conv2D: ManualConv2DLayer,
            layers.MaxPooling2D: ManualMaxPooling2DLayer,
            layers.AveragePooling2D: ManualAveragePooling2DLayer,
            layers.Flatten: ManualFlattenLayer,
            layers.GlobalAveragePooling2D: ManualGlobalAveragePooling2DLayer,
            layers.Dense: ManualDenseLayer,
            layers.Activation: None
        }

    def add_keras_layer(self, keras_layer):
        manual_layer_class = self.keras_layer_map.get(type(keras_layer))
        
        if manual_layer_class is None:
            if isinstance(keras_layer, layers.Activation):
                activation_name = keras_layer.get_config()['activation']
                if activation_name == 'relu': self.layers.append(relu)
                elif activation_name == 'softmax': self.layers.append(softmax)
                else: print(f"Warning: Activation {activation_name} not directly mapped, handled by previous/next layer or ignored.")
                return
            elif isinstance(keras_layer, layers.Dropout):
                print(f"Ignoring Dropout layer: {keras_layer.name}")
                return
            else:
                raise ValueError(f"Unsupported Keras layer type: {type(keras_layer)}")

        if type(keras_layer) == layers.Conv2D:
            weights, bias = keras_layer.get_weights()
            stride = keras_layer.strides[0]
            padding = keras_layer.padding
            manual_layer = manual_layer_class(weights, bias, stride=stride, padding=padding)
        elif type(keras_layer) in [layers.MaxPooling2D, layers.AveragePooling2D]:
            pool_size = keras_layer.pool_size
            stride = keras_layer.strides[0]
            padding = keras_layer.padding
            manual_layer = manual_layer_class(pool_size=pool_size, stride=stride, padding=padding)
        elif type(keras_layer) in [layers.Flatten, layers.GlobalAveragePooling2D]:
            manual_layer = manual_layer_class()
        elif type(keras_layer) == layers.Dense:
            weights, bias = keras_layer.get_weights()
            activation_fn = None
            activation_name = keras_layer.get_config()['activation']
            if activation_name == 'relu': activation_fn = relu
            elif activation_name == 'softmax': activation_fn = softmax
            manual_layer = manual_layer_class(weights, bias, activation_fn=activation_fn)
        else:
             raise ValueError(f"Should not happen: {type(keras_layer)}")

        self.layers.append(manual_layer)

    def load_keras_model(self, keras_model):
        self.layers = []
        for layer in keras_model.layers:
            print(f"Processing Keras layer: {layer.name} of type {type(layer)}")
            self.add_keras_layer(layer)

    def predict(self, X_batch):
        output = X_batch
        for layer in self.layers:
            if callable(layer) and not isinstance(layer, (ManualConv2DLayer, ManualMaxPooling2DLayer, ManualAveragePooling2DLayer, ManualFlattenLayer, ManualGlobalAveragePooling2DLayer, ManualDenseLayer)):
                output = layer(output)
            else:
                output = layer.forward(output)
        return output

# --- Keras Model Training and Hyperparameter Analysis ---
def load_and_preprocess_cifar10():
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Split training into train and validation (40k train, 10k val)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=10000, random_state=42, stratify=y_train_full
    )

    # Normalize pixel values
    x_train = x_train.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    num_classes = 10

    print(f"x_train shape: {x_train.shape}")
    print(f"x_val shape: {x_val.shape}")
    print(f"x_test shape: {x_test.shape}")
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), num_classes

def build_cnn_model(input_shape, num_classes, conv_layers_config, pooling_type='max', use_global_pooling=False):
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    for i, (filters, kernel_size) in enumerate(conv_layers_config):
        model.add(layers.Conv2D(filters, kernel_size, activation='relu', padding='same', name=f'conv{i+1}'))
        if pooling_type == 'max':
            model.add(layers.MaxPooling2D((2, 2), name=f'maxpool{i+1}'))
        elif pooling_type == 'avg':
            model.add(layers.AveragePooling2D((2, 2), name=f'avgpool{i+1}'))
        else:
            raise ValueError("Invalid pooling_type")

    if use_global_pooling:
        model.add(layers.GlobalAveragePooling2D(name='global_avg_pool'))
    else:
        model.add(layers.Flatten(name='flatten'))
    
    model.add(layers.Dense(128, activation='relu', name='dense1'))
    model.add(layers.Dense(num_classes, activation='softmax', name='output_dense'))
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_evaluate_cnn_variant(
    x_train, y_train, x_val, y_val, x_test, y_test, num_classes,
    conv_layers_config, pooling_type, use_global_pooling, epochs=10, batch_size=64, description=""):
    
    print(f"\n--- Training CNN: {description} ---")
    print(f"Conv Layers: {conv_layers_config}, Pooling: {pooling_type}, Global Pooling: {use_global_pooling}")

    model = build_cnn_model(x_train.shape[1:], num_classes, conv_layers_config, pooling_type, use_global_pooling)
    model.summary()
    
    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_val, y_val),
                        verbose=1) # Set to 1 or 2 for progress, 0 for silent

    plot_history(history, title_prefix=f"CNN {description}")

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    y_pred_keras_proba = model.predict(x_test)
    f1_keras = calculate_f1_macro(y_test, y_pred_keras_proba, num_classes)
    
    print(f"Test Accuracy (Keras): {test_acc:.4f}")
    print(f"Macro F1-Score (Keras): {f1_keras:.4f}")
    
    return model, f1_keras, history

def cnn_hyperparameter_analysis(x_train, y_train, x_val, y_val, x_test, y_test, num_classes):
    results = {}
    best_f1 = -1
    best_model_cnn = None
    
    print("\n=== 1. Analisis Pengaruh Jumlah Layer Konvolusi ===")
    num_conv_layer_variations = [
        [(32, (3,3))],                                  
        [(32, (3,3)), (64, (3,3))],                     
        [(32, (3,3)), (64, (3,3)), (128, (3,3))]        
    ]
    for i, config in enumerate(num_conv_layer_variations):
        desc = f"NumConvLayers_{i+1}"
        model, f1, _ = train_and_evaluate_cnn_variant(
            x_train, y_train, x_val, y_val, x_test, y_test, num_classes,
            conv_layers_config=config, pooling_type='max', use_global_pooling=False,
            epochs=5, description=desc 
        )
        results[desc] = f1
        if f1 > best_f1: best_f1 = f1; best_model_cnn = model
    print("Kesimpulan Jumlah Layer Konvolusi: [Analisis Anda di sini berdasarkan grafik dan F1-score]")

    print("\n=== 2. Analisis Pengaruh Banyak Filter ===")
    filter_variations = [
        [(16, (3,3)), (32, (3,3))], 
        [(32, (3,3)), (64, (3,3))], 
        [(64, (3,3)), (128, (3,3))] 
    ]
    for i, config in enumerate(filter_variations):
        desc = f"NumFilters_{i+1}"
        model, f1, _ = train_and_evaluate_cnn_variant(
            x_train, y_train, x_val, y_val, x_test, y_test, num_classes,
            conv_layers_config=config, pooling_type='max', use_global_pooling=False,
            epochs=5, description=desc
        )
        results[desc] = f1
        if f1 > best_f1: best_f1 = f1; best_model_cnn = model
    print("Kesimpulan Banyak Filter: [Analisis Anda di sini]")

    print("\n=== 3. Analisis Pengaruh Ukuran Filter ===")
    kernel_size_variations = [
        [(32, (2,2)), (64, (2,2))], 
        [(32, (3,3)), (64, (3,3))], 
        [(32, (5,5)), (64, (5,5))]  
    ]
    for i, config in enumerate(kernel_size_variations):
        desc = f"KernelSize_{i+1}"
        model, f1, _ = train_and_evaluate_cnn_variant(
            x_train, y_train, x_val, y_val, x_test, y_test, num_classes,
            conv_layers_config=config, pooling_type='max', use_global_pooling=False,
            epochs=5, description=desc
        )
        results[desc] = f1
        if f1 > best_f1: best_f1 = f1; best_model_cnn = model
    print("Kesimpulan Ukuran Filter: [Analisis Anda di sini]")

    print("\n=== 4. Analisis Pengaruh Jenis Pooling Layer ===")
    pooling_variations = ['max', 'avg']
    base_conv_config = [(32, (3,3)), (64, (3,3))] 
    for pool_type in pooling_variations:
        desc = f"PoolingType_{pool_type}"
        model, f1, _ = train_and_evaluate_cnn_variant(
            x_train, y_train, x_val, y_val, x_test, y_test, num_classes,
            conv_layers_config=base_conv_config, pooling_type=pool_type, use_global_pooling=False,
            epochs=5, description=desc
        )
        results[desc] = f1
        if f1 > best_f1: best_f1 = f1; best_model_cnn = model
    print("Kesimpulan Jenis Pooling: [Analisis Anda di sini]")

    print("\n--- Ringkasan F1 Scores CNN Variants ---")
    for desc, f1_val in results.items():
        print(f"{desc}: {f1_val:.4f}")
        
    if best_model_cnn:
        best_model_cnn.save("best_cnn_model.keras")
        print("\nBest Keras CNN model saved as best_cnn_model.keras")
    else:
        print("\nNo CNN model was trained successfully to be saved.")
    return best_model_cnn
