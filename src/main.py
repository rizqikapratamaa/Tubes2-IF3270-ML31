import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Import model-specific modules
import cnn as cnn_model
import rnn as rnn_model # For SimpleRNN
import lstm_1 as lstm_model # For LSTM

from utils import calculate_f1_macro

# Suppress TensorFlow eager execution warnings for cleaner output if any
# tf.compat.v1.disable_eager_execution() # Use with caution, may affect Keras behavior

def run_cnn_pipeline():
    print("========== Starting CNN Pipeline ==========")
    # 1. Load and preprocess data
    (x_train_cnn, y_train_cnn), (x_val_cnn, y_val_cnn), (x_test_cnn, y_test_cnn), num_classes_cnn = \
        cnn_model.load_and_preprocess_cifar10()

    # 2. Keras Model Training and Hyperparameter Analysis
    # This will train multiple models and save the "best" one based on F1.
    # For quick demo, epochs are low. Increase for real training.
    print("\n--- CNN Keras Hyperparameter Analysis ---")
    best_keras_cnn_model = cnn_model.cnn_hyperparameter_analysis(
        x_train_cnn, y_train_cnn, x_val_cnn, y_val_cnn, x_test_cnn, y_test_cnn, num_classes_cnn
    )
    
    if best_keras_cnn_model is None:
        print("No best Keras CNN model found from hyperparameter tuning. Building a default one.")
        # Build a default model if hyperparameter tuning didn't yield one (e.g., if skipped)
        default_conv_config = [(32, (3,3)), (64, (3,3))]
        best_keras_cnn_model = cnn_model.build_cnn_model(x_train_cnn.shape[1:], num_classes_cnn, default_conv_config)
        best_keras_cnn_model.fit(x_train_cnn, y_train_cnn, epochs=1, validation_data=(x_val_cnn, y_val_cnn), verbose=0) # Minimal fit
        best_keras_cnn_model.save("default_cnn_model.keras")
        print("Default Keras CNN model trained and saved as default_cnn_model.keras")

    # Load the best (or default) saved Keras model
    try:
        loaded_keras_cnn = keras.models.load_model("best_cnn_model.keras")
        print("\nLoaded best_cnn_model.keras for manual implementation.")
    except: # Fallback if best_cnn_model.keras doesn't exist
        try:
            loaded_keras_cnn = keras.models.load_model("default_cnn_model.keras")
            print("\nLoaded default_cnn_model.keras for manual implementation.")
        except Exception as e:
            print(f"Error loading Keras CNN model: {e}. Exiting CNN pipeline.")
            return

    # 3. Manual Forward Propagation
    print("\n--- CNN Manual Forward Propagation ---")
    cnn_manual = cnn_model.CNNFromScratch()
    cnn_manual.load_keras_model(loaded_keras_cnn)

    # 4. Test and Compare
    # Using a small subset of test data for quicker manual prediction
    # For full test, use all x_test_cnn
    test_sample_cnn = x_test_cnn[:32] # Batch of 32 for testing
    y_test_sample_cnn = y_test_cnn[:32]

    print(f"\nPredicting with Keras CNN on {test_sample_cnn.shape[0]} samples...")
    y_pred_keras_cnn_proba = loaded_keras_cnn.predict(test_sample_cnn)
    
    print(f"Predicting with Manual CNN on {test_sample_cnn.shape[0]} samples...")
    y_pred_manual_cnn_proba = cnn_manual.predict(test_sample_cnn)

    # Ensure shapes are compatible for comparison
    print(f"Keras CNN output shape: {y_pred_keras_cnn_proba.shape}")
    print(f"Manual CNN output shape: {y_pred_manual_cnn_proba.shape}")

    # Compare raw outputs (optional, for debugging)
    # print("Keras output (first 5):", y_pred_keras_cnn_proba[:5])
    # print("Manual output (first 5):", y_pred_manual_cnn_proba[:5])
    # print("Difference (sum of abs):", np.sum(np.abs(y_pred_keras_cnn_proba - y_pred_manual_cnn_proba)))
    
    f1_keras_cnn = calculate_f1_macro(y_test_sample_cnn, y_pred_keras_cnn_proba, num_classes_cnn)
    f1_manual_cnn = calculate_f1_macro(y_test_sample_cnn, y_pred_manual_cnn_proba, num_classes_cnn)

    print(f"\nMacro F1-Score (Keras CNN on test subset): {f1_keras_cnn:.4f}")
    print(f"Macro F1-Score (Manual CNN on test subset): {f1_manual_cnn:.4f}")
    
    # Full test set comparison (can be slow for manual)
    # print("\nPredicting on full test set (this might take time for manual)...")
    # y_pred_keras_full_cnn = loaded_keras_cnn.predict(x_test_cnn)
    # y_pred_manual_full_cnn = cnn_manual.predict(x_test_cnn) # This can be VERY slow
    # f1_keras_cnn_full = calculate_f1_macro(y_test_cnn, y_pred_keras_full_cnn, num_classes_cnn)
    # f1_manual_cnn_full = calculate_f1_macro(y_test_cnn, y_pred_manual_full_cnn, num_classes_cnn)
    # print(f"Macro F1-Score (Keras CNN on full test set): {f1_keras_cnn_full:.4f}")
    # print(f"Macro F1-Score (Manual CNN on full test set): {f1_manual_cnn_full:.4f}")

    print("========== CNN Pipeline Finished ==========\n")


def run_rnn_pipeline(rnn_type='SimpleRNN'):
    print(f"========== Starting {rnn_type} Pipeline ==========")
    module = rnn_model if rnn_type == 'SimpleRNN' else lstm_model
    
    # 1. Load and preprocess data (IMDB for demo)
    (x_train_rnn, y_train_rnn), (x_val_rnn, y_val_rnn), (x_test_rnn, y_test_rnn), num_classes_rnn, vocab_size_rnn = \
        module.load_and_preprocess_imdb(num_words=module.MAX_FEATURES, maxlen=module.MAX_LEN)

    # 2. Keras Model Training and Hyperparameter Analysis
    print(f"\n--- {rnn_type} Keras Hyperparameter Analysis ---")
    # The hyperparameter analysis function from rnn_model can be used for both by passing rnn_type
    best_keras_rnn_model = rnn_model.rnn_hyperparameter_analysis(
        x_train_rnn, y_train_rnn, x_val_rnn, y_val_rnn, x_test_rnn, y_test_rnn,
        num_classes_rnn, vocab_size_rnn, rnn_type_to_test=rnn_type # Pass rnn_type here
    )
    
    if best_keras_rnn_model is None:
        print(f"No best Keras {rnn_type} model found. Building a default one.")
        default_rnn_config = [64] # One layer with 64 units
        best_keras_rnn_model = module.build_rnn_model( # Use module's build_rnn_model
            vocab_size_rnn, module.EMBEDDING_DIM, module.MAX_LEN, num_classes_rnn,
            default_rnn_config, rnn_type=rnn_type, bidirectional=False
        )
        best_keras_rnn_model.fit(x_train_rnn, y_train_rnn, epochs=1, validation_data=(x_val_rnn, y_val_rnn), verbose=0)
        best_keras_rnn_model.save(f"default_{rnn_type.lower()}_model.keras")
        print(f"Default Keras {rnn_type} model trained and saved as default_{rnn_type.lower()}_model.keras")

    # Load the best (or default) saved Keras model
    try:
        loaded_keras_rnn = keras.models.load_model(f"best_{rnn_type.lower()}_model.keras")
        print(f"\nLoaded best_{rnn_type.lower()}_model.keras for manual implementation.")
    except:
        try:
            loaded_keras_rnn = keras.models.load_model(f"default_{rnn_type.lower()}_model.keras")
            print(f"\nLoaded default_{rnn_type.lower()}_model.keras for manual implementation.")
        except Exception as e:
            print(f"Error loading Keras {rnn_type} model: {e}. Exiting {rnn_type} pipeline.")
            return

    # 3. Manual Forward Propagation
    print(f"\n--- {rnn_type} Manual Forward Propagation ---")
    if rnn_type == 'SimpleRNN':
        rnn_manual = module.SimpleRNNFromScratch()
    elif rnn_type == 'LSTM':
        rnn_manual = module.LSTMFromScratch()
    else:
        raise ValueError("Invalid rnn_type")
        
    rnn_manual.load_keras_model(loaded_keras_rnn)

    # 4. Test and Compare
    test_sample_rnn = x_test_rnn[:32] 
    y_test_sample_rnn = y_test_rnn[:32]

    print(f"\nPredicting with Keras {rnn_type} on {test_sample_rnn.shape[0]} samples...")
    y_pred_keras_rnn_proba = loaded_keras_rnn.predict(test_sample_rnn)
    
    print(f"Predicting with Manual {rnn_type} on {test_sample_rnn.shape[0]} samples...")
    y_pred_manual_rnn_proba = rnn_manual.predict(test_sample_rnn)

    print(f"Keras {rnn_type} output shape: {y_pred_keras_rnn_proba.shape}")
    print(f"Manual {rnn_type} output shape: {y_pred_manual_rnn_proba.shape}")

    # Adjust for binary (IMDB) vs multiclass F1 calculation
    if num_classes_rnn == 2:
        y_pred_keras_rnn_classes = (y_pred_keras_rnn_proba > 0.5).astype(int)
        y_pred_manual_rnn_classes = (y_pred_manual_rnn_proba > 0.5).astype(int)
        f1_keras_rnn = calculate_f1_macro(y_test_sample_rnn, y_pred_keras_rnn_classes, num_classes_rnn)
        f1_manual_rnn = calculate_f1_macro(y_test_sample_rnn, y_pred_manual_rnn_classes, num_classes_rnn)
    else: # Multiclass
        f1_keras_rnn = calculate_f1_macro(y_test_sample_rnn, y_pred_keras_rnn_proba, num_classes_rnn)
        f1_manual_rnn = calculate_f1_macro(y_test_sample_rnn, y_pred_manual_rnn_proba, num_classes_rnn)

    print(f"\nMacro F1-Score (Keras {rnn_type} on test subset): {f1_keras_rnn:.4f}")
    print(f"Macro F1-Score (Manual {rnn_type} on test subset): {f1_manual_rnn:.4f}")

    print(f"========== {rnn_type} Pipeline Finished ==========\n")


if __name__ == "__main__":
    # --- CNN Pipeline ---
    # run_cnn_pipeline()
    
    # --- Simple RNN Pipeline ---
    # Note: For RNN/LSTM, Keras training can take a while.
    # The rnn_hyperparameter_analysis function uses few epochs for speed.
    # run_rnn_pipeline(rnn_type='SimpleRNN')

    # --- LSTM Pipeline ---
    run_rnn_pipeline(rnn_type='LSTM')

    print("All pipelines completed.")
