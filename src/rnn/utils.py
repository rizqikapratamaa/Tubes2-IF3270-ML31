import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

def relu(x):
    return np.maximum(0, x)

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def sigmoid(x):
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))

def tanh(x):
    return np.tanh(x)

def d_relu_dz(z):
    """Derivative of ReLU w.r.t. its input z."""
    return (z > 0).astype(float)

def d_sigmoid_dz(sigmoid_output):
    """Derivative of sigmoid w.r.t. its input z, given sigmoid_output = sigmoid(z)."""
    return sigmoid_output * (1 - sigmoid_output)

def d_tanh_dz(tanh_output):
    """Derivative of tanh w.r.t. its input z, given tanh_output = tanh(z)."""
    return 1 - np.square(tanh_output)

class ManualSparseCategoricalCrossentropy:
    def calculate(self, y_pred_proba, y_true_labels):
        """
        Calculates sparse categorical crossentropy loss.
        y_pred_proba: (N, C) array of probabilities.
        y_true_labels: (N,) or (N,1) array of integer labels.
        """
        N = y_true_labels.shape[0]
        epsilon = 1e-9
        correct_logprobs = -np.log(y_pred_proba[np.arange(N), y_true_labels.flatten()] + epsilon)
        loss = np.sum(correct_logprobs) / N
        return loss

    def derivative(self, y_pred_proba, y_true_labels):
        """
        Calculates the derivative of sparse categorical crossentropy loss
        w.r.t. the pre-softmax logits (often denoted as Z), assuming y_pred_proba is the output of softmax.
        The derivative dL/dZ = y_pred_proba - y_true_one_hot.
        y_pred_proba: (N, C) array of probabilities (output of softmax).
        y_true_labels: (N,) or (N,1) array of integer labels.
        """
        N = y_true_labels.shape[0]
        num_classes = y_pred_proba.shape[1]
        
        y_true_one_hot = np.zeros_like(y_pred_proba)
        y_true_one_hot[np.arange(N), y_true_labels.flatten()] = 1
        
        dL_dZ = (y_pred_proba - y_true_one_hot) / N 
        return dL_dZ

def calculate_f1_macro(y_true, y_pred_proba, num_classes=None):
    if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] > 1:
        y_pred = np.argmax(y_pred_proba, axis=1)
    elif y_pred_proba.ndim == 1 or y_pred_proba.shape[1] == 1:
        if y_pred_proba.dtype == float:
             y_pred = (y_pred_proba > 0.5).astype(int)
        else:
             y_pred = y_pred_proba.astype(int)
    else:
        y_pred = y_pred_proba.astype(int)

    if y_true.ndim == 2 and y_true.shape[1] > 1:
        y_true_labels = np.argmax(y_true, axis=1)
    else:
        y_true_labels = y_true.flatten().astype(int)

    return f1_score(y_true_labels, y_pred, average='macro', zero_division=0)

def plot_history(history, title_prefix=""):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{title_prefix} Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    acc_key = 'accuracy'
    val_acc_key = 'val_accuracy'
    if 'sparse_categorical_accuracy' in history.history:
        acc_key = 'sparse_categorical_accuracy'
        val_acc_key = 'val_sparse_categorical_accuracy'
    elif 'binary_accuracy' in history.history:
        acc_key = 'binary_accuracy'
        val_acc_key = 'val_binary_accuracy'
    
    if acc_key in history.history and val_acc_key in history.history:
        plt.plot(history.history[acc_key], label='Training Accuracy')
        plt.plot(history.history[val_acc_key], label='Validation Accuracy')
        plt.title(f'{title_prefix} Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
    else:
        print(f"Accuracy metrics ({acc_key}, {val_acc_key}) not found in history for plotting.")
        
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(1)
