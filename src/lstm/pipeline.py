import numpy as np
import tensorflow as tf
from tensorflow import keras

import lstm as lstm_model 

from utils import calculate_f1_macro

def run_lstm_pipeline(rnn_type='LSTM'):
    print(f"========== Starting {rnn_type} Pipeline ==========")
    
    best_model_path = f"best_{rnn_type.lower()}_model.keras"
    default_model_path = f"default_{rnn_type.lower()}_model.keras"

    module = lstm_model
    print("Loading NusaX Sentiment data for LSTM...")
    (x_train_rnn, y_train_rnn), (x_val_rnn, y_val_rnn), (x_test_rnn, y_test_rnn), \
    num_classes_rnn, vocab_size_rnn, _ = module.load_and_preprocess_nusax_sentiment(
            max_features=module.MAX_FEATURES, maxlen=module.MAX_LEN # Use constants from lstm_model
    )
    embedding_dim_rnn = module.EMBEDDING_DIM
    maxlen_rnn = module.MAX_LEN

    print(f"\n--- {rnn_type} Keras Hyperparameter Analysis ---")
    best_keras_rnn_model = module.lstm_hyperparameter_analysis(
        x_train_rnn, y_train_rnn, x_val_rnn, y_val_rnn, x_test_rnn, y_test_rnn,
        num_classes_rnn, vocab_size_rnn, embedding_dim_rnn, maxlen_rnn, analysis_epochs=1 # Short epochs
    )

    if best_keras_rnn_model is None:
        print(f"No best Keras {rnn_type} model. Building a default one.")
        default_lstm_config = [{'units': 64, 'bidirectional': True}] # Example config
        best_keras_rnn_model = module.build_lstm_model(
            vocab_size_rnn, embedding_dim_rnn, maxlen_rnn, num_classes_rnn, default_lstm_config
        )
        best_keras_rnn_model.fit(x_train_rnn, y_train_rnn, epochs=1, validation_data=(x_val_rnn, y_val_rnn), verbose=0)
        best_keras_rnn_model.save(default_model_path)
        print(f"Default Keras {rnn_type} model saved as {default_model_path}")

    manual_model_constructor = module.LSTMFromScratch

    try:
        loaded_keras_rnn = keras.models.load_model(best_model_path)
        print(f"\nLoaded {best_model_path} for manual implementation.")
    except IOError:
        try:
            loaded_keras_rnn = keras.models.load_model(default_model_path)
            print(f"\nLoaded {default_model_path} for manual implementation.")
        except Exception as e:
            print(f"Error loading Keras {rnn_type} model: {e}. Exiting {rnn_type} pipeline.")
            return

    # Manual Forward Propagation
    print(f"\n--- {rnn_type} Manual Forward Propagation ---")
    rnn_manual = manual_model_constructor()
    rnn_manual.load_keras_model(loaded_keras_rnn)

    test_sample_rnn = x_test_rnn[:32] 
    y_test_sample_rnn = y_test_rnn[:32]

    print(f"\nPredicting with Keras {rnn_type} on {test_sample_rnn.shape[0]} samples...")
    y_pred_keras_rnn_proba = loaded_keras_rnn.predict(test_sample_rnn)
    
    print(f"Predicting with Manual {rnn_type} on {test_sample_rnn.shape[0]} samples...")
    y_pred_manual_rnn_proba = rnn_manual.predict(test_sample_rnn)

    f1_keras_rnn = calculate_f1_macro(y_test_sample_rnn, y_pred_keras_rnn_proba)
    f1_manual_rnn = calculate_f1_macro(y_test_sample_rnn, y_pred_manual_rnn_proba)

    print(f"\nMacro F1-Score (Keras {rnn_type} on test subset): {f1_keras_rnn:.4f}")
    print(f"Macro F1-Score (Manual {rnn_type} on test subset): {f1_manual_rnn:.4f}")
    print(f"Sum of absolute differences ({rnn_type}): {np.sum(np.abs(y_pred_keras_rnn_proba - y_pred_manual_rnn_proba)):.6f}")
    
    if rnn_type == 'LSTM' and hasattr(module, 'ManualSparseCategoricalCrossentropy') and hasattr(module, 'ManualSGD'):
        print(f"\n--- {rnn_type} Manual Backward Propagation & Training Test (Bonus) ---")
        manual_lstm_train_test = module.LSTMFromScratch()
        manual_lstm_train_test.load_keras_model(loaded_keras_rnn)

        loss_func = module.ManualSparseCategoricalCrossentropy()
        optimizer = module.ManualSGD(learning_rate=0.01)
        manual_lstm_train_test.compile_manual(loss_fn_instance=loss_func, optimizer_instance=optimizer)

        print(f"Starting manual training with {rnn_type}FromScratch...")
        x_train_tiny = x_train_rnn[:64]
        y_train_tiny = y_train_rnn[:64]
        try:
            manual_lstm_train_test.fit_manual(x_train_tiny, y_train_tiny, epochs=1, batch_size=16)
            print("Manual training epoch completed.")
            
            y_pred_manual_after_train = manual_lstm_train_test.predict(test_sample_rnn)
            f1_manual_after_train = calculate_f1_macro(y_test_sample_rnn, y_pred_manual_after_train)
            print(f"F1 Manual ({rnn_type} on test subset) after 1 epoch of manual training: {f1_manual_after_train:.4f}")

        except NotImplementedError as e:
            print(f"Could not run manual {rnn_type} training (NotImplementedError): {e}")
        except Exception as e:
            print(f"An error occurred during manual {rnn_type} training test: {e}")
            import traceback
            traceback.print_exc()

    print(f"========== {rnn_type} Pipeline Finished ==========\n")