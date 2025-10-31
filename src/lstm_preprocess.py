# src/lstm_preprocess.py
import numpy as np
from sklearn.preprocessing import StandardScaler

def create_sequences_lstm(data, sequence_length=50):
    """
    Creates sequences (samples) for LSTM training.
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def prepare_lstm_data(df, feature_column='cpu_usage', sequence_length=50, test_ratio=0.2):
    """
    Prepares data for LSTM training and testing.
    """
    print("Starting data preparation...")
    
    # Use the raw CPU values for the LSTM
    data = df[feature_column].values
    print(f"Original data shape: {data.shape}")

    # Split into train and test based on time (DO NOT shuffle time series data!)
    split_idx = int(len(data) * (1 - test_ratio))
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    print(f"Train data shape before scaling: {train_data.shape}")
    print(f"Test data shape before scaling: {test_data.shape}")

    # Scale the data (fit only on training data to avoid data leakage)
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
    test_data_scaled = scaler.transform(test_data.reshape(-1, 1)).flatten()

    print(f"Train data shape after scaling: {train_data_scaled.shape}")

    # Create sequences for training and testing
    X_train, y_train = create_sequences_lstm(train_data_scaled, sequence_length)
    X_test, y_test = create_sequences_lstm(test_data_scaled, sequence_length)

    print(f"X_train shape before reshaping: {X_train.shape}")
    print(f"X_test shape before reshaping: {X_test.shape}")

    # CRITICAL FIX: Reshape to add feature dimension for LSTM
    # LSTM expects: (samples, timesteps, features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # PyTorch expects float32
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    print(f"Final X_train shape: {X_train.shape}")
    print(f"Final y_train shape: {y_train.shape}")
    print(f"Final X_test shape: {X_test.shape}")
    print(f"Final y_test shape: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test, scaler, split_idx