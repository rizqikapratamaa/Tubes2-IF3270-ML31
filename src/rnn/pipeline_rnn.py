from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from rnn import load_and_preprocess_nusax_sentiment, SimpleRNNFromScratch
from tensorflow.keras.optimizers import Adam

MAX_LEN = 200
MAX_FEATURES = 30000
EMBEDDING_DIM = 256

def run_rnn_pipeline(rnn_type='SimpleRNN'):
    print(f"========== Starting {rnn_type} Pipeline ==========")
    if rnn_type == 'SimpleRNN':
        (x_train_rnn, y_train_rnn), (x_val_rnn, y_val_rnn), (x_test_rnn, y_test_rnn), num_classes_rnn, vocab_size_rnn, text_vectorizer = \
            load_and_preprocess_nusax_sentiment(max_features=MAX_FEATURES, maxlen=MAX_LEN)
    else:
        raise NotImplementedError('Only SimpleRNN pipeline is supported in this file.')
    print(f"\n--- {rnn_type} Keras Hyperparameter Analysis ---")
    rnn_config = [
        {'units': 128, 'bidirectional': True, 'return_sequences': True},
        {'units': 128, 'bidirectional': True, 'return_sequences': True},
        {'units': 128, 'bidirectional': True, 'return_sequences': False}
    ]
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    from rnn import build_simplernn_model
    model = build_simplernn_model(
        vocab_size_rnn, EMBEDDING_DIM, MAX_LEN, num_classes_rnn, rnn_config, dropout_rate=0.5
    )
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print(model.summary())
    history = model.fit(
        x_train_rnn, y_train_rnn,
        validation_data=(x_val_rnn, y_val_rnn),
        epochs=30,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    test_loss, test_acc = model.evaluate(x_test_rnn, y_test_rnn, verbose=1)
    print(f"Test Accuracy: {test_acc:.4f}")
    model.save("best_simplernn_model.keras")
    print("Best Keras SimpleRNN model saved as best_simplernn_model.keras")
    try:
        loaded_keras_rnn = keras.models.load_model("best_simplernn_model.keras")
        print(f"\nLoaded best_simplernn_model.keras for manual implementation.")
    except:
        print(f"Error loading Keras {rnn_type} model. Exiting {rnn_type} pipeline.")
        return
    print(f"\n--- {rnn_type} Manual Forward Propagation ---")
    rnn_manual = SimpleRNNFromScratch()
    rnn_manual.build_from_keras(loaded_keras_rnn)
    test_sample_rnn = x_test_rnn[:32]
    y_test_sample_rnn = y_test_rnn[:32]
    print(f"\nPredicting with Keras {rnn_type} on {test_sample_rnn.shape[0]} samples...")
    y_pred_keras_rnn_proba = loaded_keras_rnn.predict(test_sample_rnn)
    print(f"Predicting with Manual {rnn_type} on {test_sample_rnn.shape[0]} samples...")
    y_pred_manual_rnn_proba = rnn_manual.predict(test_sample_rnn)
    print(f"Keras {rnn_type} output shape: {y_pred_keras_rnn_proba.shape}")
    print(f"Manual {rnn_type} output shape: {y_pred_manual_rnn_proba.shape}")
    from utils import calculate_f1_macro
    f1_keras_rnn = calculate_f1_macro(y_test_sample_rnn, y_pred_keras_rnn_proba, num_classes_rnn)
    f1_manual_rnn = calculate_f1_macro(y_test_sample_rnn, y_pred_manual_rnn_proba, num_classes_rnn)
    print(f"\nMacro F1-Score (Keras {rnn_type} on test subset): {f1_keras_rnn:.4f}")
    print(f"Macro F1-Score (Manual {rnn_type} on test subset): {f1_manual_rnn:.4f}")
    print(f"========== {rnn_type} Pipeline Finished ==========")

if __name__ == "__main__":
    run_rnn_pipeline(rnn_type='SimpleRNN')
    print("All SimpleRNN pipelines completed.")
