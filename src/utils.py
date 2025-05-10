import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def pad_with_zeros(X, pad_h, pad_w):
    """Pads X with zeros. X is (N, H, W, C)."""
    if pad_h == 0 and pad_w == 0:
        return X
    return np.pad(X, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant', constant_values=0)

def calculate_f1_macro(y_true, y_pred_proba, num_classes):
    from sklearn.metrics import f1_score
    if y_pred_proba.ndim == 2: 
        y_pred = np.argmax(y_pred_proba, axis=1)
    else: 
        y_pred = y_pred_proba

    if y_true.ndim == 2 and y_true.shape[1] > 1: 
        y_true_labels = np.argmax(y_true, axis=1)
    else:
        y_true_labels = y_true.flatten() 

    return f1_score(y_true_labels, y_pred, average='macro', zero_division=0)

def plot_history(history, title_prefix=""):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{title_prefix} Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    if 'accuracy' in history.history and 'val_accuracy' in history.history:
        acc_metric = 'accuracy'
        val_acc_metric = 'val_accuracy'
    elif 'sparse_categorical_accuracy' in history.history and 'val_sparse_categorical_accuracy' in history.history:
        acc_metric = 'sparse_categorical_accuracy'
        val_acc_metric = 'val_sparse_categorical_accuracy'
    else: 
        print("Accuracy metrics not found in history for plotting.")
        plt.tight_layout()
        plt.show()
        return

    plt.plot(history.history[acc_metric], label='Training Accuracy')
    plt.plot(history.history[val_acc_metric], label='Validation Accuracy')
    plt.title(f'{title_prefix} Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
