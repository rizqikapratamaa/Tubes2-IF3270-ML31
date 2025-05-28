import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils import relu, softmax, calculate_f1_macro, plot_history
from sklearn.model_selection import train_test_split

# convolution layer
class ManualConv2DLayer:
    def __init__(self, kernel, bias, stride=(1,1), padding='valid'): # stride is tuple (sH, sW)
        self.kernel = kernel  # Shape: (KH, KW, C_in, C_out)
        self.bias = bias  # Shape: (C_out,)
        self.padding = padding.lower()

        if isinstance(stride, int):
            self.stride_h, self.stride_w = stride, stride
        elif isinstance(stride, tuple) and len(stride) == 2:
            self.stride_h, self.stride_w = stride[0], stride[1]
        else:
            raise ValueError(f"Invalid stride format for Conv2D: {stride}")


    def forward(self, X_batch):  # X_batch shape: (N, H_in, W_in, C_in)
        N, H_in, W_in, C_in = X_batch.shape
        KH, KW, _, C_out = self.kernel.shape

        if self.padding == 'same':
            out_height = int(np.ceil(float(H_in) / float(self.stride_h)))
            out_width = int(np.ceil(float(W_in) / float(self.stride_w)))
            
            pad_h_total = max((out_height - 1) * self.stride_h + KH - H_in, 0)
            pad_w_total = max((out_width - 1) * self.stride_w + KW - W_in, 0)
            
            pad_top = pad_h_total // 2
            pad_bottom = pad_h_total - pad_top
            pad_left = pad_w_total // 2
            pad_right = pad_w_total - pad_left
            
            X_padded = np.pad(X_batch, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0.0)
        elif self.padding == 'valid':
            out_height = (H_in - KH) // self.stride_h + 1
            out_width = (W_in - KW) // self.stride_w + 1
            X_padded = X_batch
        else:
            raise ValueError(f"Unsupported padding type: {self.padding}")

        if out_height <= 0 or out_width <= 0:
             raise ValueError(f"Output dimensions non-positive: H={out_height}, W={out_width}. Input: H_in={H_in}, W_in={W_in}, Kernel: KH={KH}, KW={KW}, Stride: ({self.stride_h},{self.stride_w}), Padding: {self.padding}")

        shape_col = (N, out_height, out_width, KH, KW, C_in)
        
        strides_col = (
            X_padded.strides[0],
            self.stride_h * X_padded.strides[1],
            self.stride_w * X_padded.strides[2],
            X_padded.strides[1],
            X_padded.strides[2],
            X_padded.strides[3]
        )
        
        H_padded, W_padded = X_padded.shape[1:3]
        if (out_height > 0 and (out_height - 1) * self.stride_h + KH > H_padded) or \
           (out_width > 0 and (out_width - 1) * self.stride_w + KW > W_padded):
            raise ValueError(
                f"Calculated patch extent exceeds padded input dimensions. "
                f"H_padded={H_padded}, W_padded={W_padded}. "
                f"out_height={out_height}, out_width={out_width}. "
                f"Required H: {(out_height - 1) * self.stride_h + KH}, "
                f"Required W: {(out_width - 1) * self.stride_w + KW}."
            )

        X_col = np.lib.stride_tricks.as_strided(X_padded, shape=shape_col, strides=strides_col)
        
        X_col_reshaped = X_col.reshape(N * out_height * out_width, KH * KW * C_in)
        kernel_reshaped = self.kernel.reshape(KH * KW * C_in, C_out)
        
        output_reshaped = np.dot(X_col_reshaped, kernel_reshaped)
        output = output_reshaped.reshape(N, out_height, out_width, C_out)
        
        output += self.bias
        
        return output

# max pooling layer
class ManualMaxPooling2DLayer:
    def __init__(self, pool_size=(2, 2), stride=None, padding='valid'):
        self.pool_size = pool_size # (PH, PW)
        self.padding = padding.lower()
        if stride is None:
            self.stride_h, self.stride_w = pool_size[0], pool_size[1]
        elif isinstance(stride, int):
            self.stride_h, self.stride_w = stride, stride
        elif isinstance(stride, tuple) and len(stride) == 2:
            self.stride_h, self.stride_w = stride[0], stride[1]
        else:
            raise ValueError(f"Invalid stride format for MaxPooling: {stride}")

    def forward(self, X_batch):  # X_batch shape: (N, H_in, W_in, C_in)
        N, H_in, W_in, C_in = X_batch.shape
        PH, PW = self.pool_size

        if self.padding == 'same':
            out_height = int(np.ceil(float(H_in) / float(self.stride_h)))
            out_width = int(np.ceil(float(W_in) / float(self.stride_w)))
            pad_h_total = max((out_height - 1) * self.stride_h + PH - H_in, 0)
            pad_w_total = max((out_width - 1) * self.stride_w + PW - W_in, 0)
            pad_top = pad_h_total // 2; pad_bottom = pad_h_total - pad_top
            pad_left = pad_w_total // 2; pad_right = pad_w_total - pad_left

            X_float = X_batch if np.issubdtype(X_batch.dtype, np.floating) else X_batch.astype(np.float32)
            X_padded = np.pad(X_float, ((0,0),(pad_top,pad_bottom),(pad_left,pad_right),(0,0)), 
                              mode='constant', constant_values=-np.inf)
        elif self.padding == 'valid':
            out_height = (H_in - PH) // self.stride_h + 1
            out_width = (W_in - PW) // self.stride_w + 1
            if not np.issubdtype(X_batch.dtype, np.floating):
                 X_padded = X_batch.astype(np.float32) 
            else:
                 X_padded = X_batch
        else:
            raise ValueError(f"Unsupported padding: {self.padding}")

        if out_height <= 0 or out_width <= 0:
            raise ValueError(f"Invalid output dims: H_out={out_height}, W_out={out_width} for H_in={H_in}, W_in={W_in}, Pool=({PH},{PW}), Stride=({self.stride_h},{self.stride_w})")
        
        S_N, S_H, S_W, S_C = X_padded.strides
        output_shape_strided = (N, out_height, out_width, C_in, PH, PW)
        output_strides = (S_N, self.stride_h * S_H, self.stride_w * S_W, S_C, S_H, S_W)

        H_padded, W_padded = X_padded.shape[1:3]
        if (out_height > 0 and (out_height - 1) * self.stride_h + PH > H_padded) or \
           (out_width > 0 and (out_width - 1) * self.stride_w + PW > W_padded):
            raise ValueError(f"Patch size ({PH},{PW}) with stride ({self.stride_h},{self.stride_w}) exceeds padded input ({H_padded},{W_padded}) for output ({out_height},{out_width})")
        
        patches = np.lib.stride_tricks.as_strided(X_padded, shape=output_shape_strided, strides=output_strides)
        output = np.max(patches, axis=(4, 5))
        return output

# average pooling layer
class ManualAveragePooling2DLayer:
    def __init__(self, pool_size=(2, 2), stride=None, padding='valid'):
        self.pool_size = pool_size
        self.padding = padding.lower()
        if stride is None:
            self.stride_h, self.stride_w = pool_size[0], pool_size[1]
        elif isinstance(stride, int):
            self.stride_h, self.stride_w = stride, stride
        elif isinstance(stride, tuple) and len(stride) == 2:
            self.stride_h, self.stride_w = stride[0], stride[1]
        else:
            raise ValueError(f"Invalid stride format for AveragePooling: {stride}")


    def forward(self, X_batch):  # X_batch shape: (N, H_in, W_in, C_in)
        N, H_in, W_in, C_in = X_batch.shape
        PH, PW = self.pool_size

        if self.padding == 'same':
            out_height = int(np.ceil(float(H_in) / float(self.stride_h)))
            out_width = int(np.ceil(float(W_in) / float(self.stride_w)))
            pad_h_total = max((out_height - 1) * self.stride_h + PH - H_in, 0)
            pad_w_total = max((out_width - 1) * self.stride_w + PW - W_in, 0)
            pad_top = pad_h_total // 2; pad_bottom = pad_h_total - pad_top
            pad_left = pad_w_total // 2; pad_right = pad_w_total - pad_left
            X_padded = np.pad(X_batch, ((0,0),(pad_top,pad_bottom),(pad_left,pad_right),(0,0)),
                              mode='constant', constant_values=0.0)
        elif self.padding == 'valid':
            out_height = (H_in - PH) // self.stride_h + 1
            out_width = (W_in - PW) // self.stride_w + 1
            X_padded = X_batch
        else:
            raise ValueError(f"Unsupported padding: {self.padding}")

        if out_height <= 0 or out_width <= 0:
            raise ValueError(f"Invalid output dims: H_out={out_height}, W_out={out_width} for H_in={H_in}, W_in={W_in}, Pool=({PH},{PW}), Stride=({self.stride_h},{self.stride_w})")

        S_N, S_H, S_W, S_C = X_padded.strides
        output_shape_strided = (N, out_height, out_width, C_in, PH, PW)
        output_strides = (S_N, self.stride_h * S_H, self.stride_w * S_W, S_C, S_H, S_W)
        
        H_padded, W_padded = X_padded.shape[1:3]
        if (out_height > 0 and (out_height - 1) * self.stride_h + PH > H_padded) or \
           (out_width > 0 and (out_width - 1) * self.stride_w + PW > W_padded):
            raise ValueError(f"Patch size ({PH},{PW}) with stride ({self.stride_h},{self.stride_w}) exceeds padded input ({H_padded},{W_padded}) for output ({out_height},{out_width})")

        patches = np.lib.stride_tricks.as_strided(X_padded, shape=output_shape_strided, strides=output_strides)
        output = np.mean(patches, axis=(4, 5))
        return output

# flatten layer
class ManualFlattenLayer:
    def forward(self, X_batch):  # X_batch shape: (N, H, W, C)
        N = X_batch.shape[0]
        return X_batch.reshape(N, -1)  # Reshape to (N, H*W*C)

# global average pooling layer
class ManualGlobalAveragePooling2DLayer:
    def forward(self, X_batch):  # X_batch shape: (N, H, W, C)
        return np.mean(X_batch, axis=(1, 2))  # Result shape (N, C)

# dense layer
class ManualDenseLayer:
    def __init__(self, weights, bias, activation_fn=None):
        self.weights = weights  # Shape: (input_features, output_features)
        self.bias = bias  # Shape: (output_features,)
        self.activation_fn = activation_fn

    def forward(self, X_batch):  # X_batch shape: (N, input_features)
        output = np.dot(X_batch, self.weights) + self.bias
        if self.activation_fn:
            output = self.activation_fn(output)
        return output

# self made CNN
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
        
        layer_activation_fn = None
        if hasattr(keras_layer, 'activation') and keras_layer.activation is not None:
            try:
                activation_name = keras.activations.serialize(keras_layer.activation)
                if activation_name == 'relu': layer_activation_fn = relu
                elif activation_name == 'softmax': layer_activation_fn = softmax
                # TODO: add more activation layer
            except:
                pass


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
            stride_param = keras_layer.strides 
            padding = keras_layer.padding
            manual_layer = manual_layer_class(weights, bias, stride=stride_param, padding=padding)
        elif type(keras_layer) in [layers.MaxPooling2D, layers.AveragePooling2D]:
            pool_size = keras_layer.pool_size
            stride_param = keras_layer.strides 
            padding = keras_layer.padding
            manual_layer = manual_layer_class(pool_size=pool_size, stride=stride_param, padding=padding)
        elif type(keras_layer) in [layers.Flatten, layers.GlobalAveragePooling2D]:
            manual_layer = manual_layer_class()
        elif type(keras_layer) == layers.Dense:
            weights, bias = keras_layer.get_weights()
            
            current_dense_activation_fn = None
            activation_name_dense = keras_layer.get_config()['activation']
            if activation_name_dense == 'relu': current_dense_activation_fn = relu
            elif activation_name_dense == 'softmax': current_dense_activation_fn = softmax
            
            manual_layer = manual_layer_class(weights, bias, activation_fn=current_dense_activation_fn)
            layer_activation_fn = None
        else:
             raise ValueError(f"Layer type {type(keras_layer)} was mapped but not handled in conditional block.")

        self.layers.append(manual_layer)
        
        if layer_activation_fn and not (type(keras_layer) == layers.Dense):
            self.layers.append(layer_activation_fn)
            print(f"Added separate activation {layer_activation_fn.__name__} after {keras_layer.name}")

    def load_keras_model(self, keras_model):
        self.layers = [] # reset layers
        for layer in keras_model.layers:
            print(f"Processing Keras layer: {layer.name} of type {type(layer)}")
            self.add_keras_layer(layer)

    def predict(self, X_batch):
        output = X_batch
        for layer_idx, layer in enumerate(self.layers):
            if callable(layer) and not isinstance(layer, (ManualConv2DLayer, ManualMaxPooling2DLayer, ManualAveragePooling2DLayer, ManualFlattenLayer, ManualGlobalAveragePooling2DLayer, ManualDenseLayer)):
                output = layer(output)
            elif hasattr(layer, 'forward'):
                output = layer.forward(output)
            else:
                raise TypeError(f"Layer {layer_idx} is of unhandled type: {type(layer)}")
        return output

# cifar10 loading
def load_and_preprocess_cifar10():
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=10000, random_state=42, stratify=y_train_full
    )

    x_train = x_train.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    num_classes = 10

    print(f"x_train shape: {x_train.shape}")
    print(f"x_val shape: {x_val.shape}")
    print(f"x_test shape: {x_test.shape}")
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), num_classes

# build keras CNN model
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
    
    # keras model training and evaluation
    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_val, y_val),
                        verbose=1) 

    plot_history(history, title_prefix=f"CNN {description} Keras")

    test_loss_keras, test_acc_keras = model.evaluate(x_test, y_test, verbose=0)
    y_pred_keras_proba = model.predict(x_test)
    f1_keras = calculate_f1_macro(y_test, y_pred_keras_proba, num_classes)
    
    print(f"Test Accuracy (Keras): {test_acc_keras:.4f}")
    print(f"Macro F1-Score (Keras): {f1_keras:.4f}")

    return model, f1_keras, history

def cnn_hyperparameter_analysis(x_train, y_train, x_val, y_val, x_test, y_test, num_classes):
    results = {}
    best_f1 = -1
    best_model_cnn = None
    
    analysis_epochs = 10

    print("\n=== 1. Analisis Pengaruh Jumlah Layer Konvolusi ===")
    num_conv_layer_variations = [
        [(32, (3,3))],                                  
        [(32, (3,3)), (64, (3,3))],                     
        [(32, (3,3)), (64, (3,3)), (128, (3,3))]
    ]
    for i, config in enumerate(num_conv_layer_variations):
        desc = f"NumConvLayers_{i+1}"
        keras_model_variant, f1_score_keras, _ = train_and_evaluate_cnn_variant(
            x_train, y_train, x_val, y_val, x_test, y_test, num_classes,
            conv_layers_config=config, pooling_type='max', use_global_pooling=False,
            epochs=analysis_epochs, description=desc 
        )
        results[desc] = f1_score_keras
        if f1_score_keras > best_f1:
            best_f1 = f1_score_keras
            best_model_cnn = keras_model_variant
    print("Kesimpulan Jumlah Layer Konvolusi: [Analisis Anda di sini berdasarkan grafik dan F1-score]")


    print("\n=== 2. Analisis Pengaruh Banyak Filter ===")
    filter_variations = [
        [(16, (3,3)), (32, (3,3))], 
        [(32, (3,3)), (64, (3,3))], 
        [(64, (3,3)), (128, (3,3))] 
    ]
    for i, config in enumerate(filter_variations):
        desc = f"NumFilters_{i+1}"
        keras_model_variant, f1_score_keras, _ = train_and_evaluate_cnn_variant(
            x_train, y_train, x_val, y_val, x_test, y_test, num_classes,
            conv_layers_config=config, pooling_type='max', use_global_pooling=False,
            epochs=analysis_epochs, description=desc
        )
        results[desc] = f1_score_keras
        if f1_score_keras > best_f1: best_f1 = f1_score_keras; best_model_cnn = keras_model_variant
    print("Kesimpulan Banyak Filter: [Analisis Anda di sini]")

    print("\n=== 3. Analisis Pengaruh Ukuran Filter ===")
    kernel_size_variations = [
        [(32, (2,2)), (64, (2,2))], 
        [(32, (3,3)), (64, (3,3))], 
        [(32, (5,5)), (64, (5,5))]  
    ]
    for i, config in enumerate(kernel_size_variations):
        desc = f"KernelSize_{i+1}"
        keras_model_variant, f1_score_keras, _ = train_and_evaluate_cnn_variant(
            x_train, y_train, x_val, y_val, x_test, y_test, num_classes,
            conv_layers_config=config, pooling_type='max', use_global_pooling=False,
            epochs=analysis_epochs, description=desc
        )
        results[desc] = f1_score_keras
        if f1_score_keras > best_f1: best_f1 = f1_score_keras; best_model_cnn = keras_model_variant
    print("Kesimpulan Ukuran Filter: [Analisis Anda di sini]")

    print("\n=== 4. Analisis Pengaruh Jenis Pooling Layer ===")
    pooling_variations = ['max', 'avg']
    base_conv_config = [(32, (3,3)), (64, (3,3))] 
    for pool_type in pooling_variations:
        desc = f"PoolingType_{pool_type}"
        keras_model_variant, f1_score_keras, _ = train_and_evaluate_cnn_variant(
            x_train, y_train, x_val, y_val, x_test, y_test, num_classes,
            conv_layers_config=base_conv_config, pooling_type=pool_type, use_global_pooling=False,
            epochs=analysis_epochs, description=desc
        )
        results[desc] = f1_score_keras
        if f1_score_keras > best_f1: best_f1 = f1_score_keras; best_model_cnn = keras_model_variant
    print("Kesimpulan Jenis Pooling: [Analisis Anda di sini]")


    print("\n--- Ringkasan F1 Scores CNN Variants (Keras Model) ---")
    for desc, f1_val in results.items():
        print(f"{desc}: {f1_val:.4f}")
        
    if best_model_cnn:
        best_model_cnn.save("best_cnn_model.keras")
        print("\nBest Keras CNN model saved as best_cnn_model.keras")
    else:
        print("\nNo Keras CNN model was trained successfully to be saved (or analysis was too short).")
    return best_model_cnn
