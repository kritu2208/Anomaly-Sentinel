# scripts/03_train_lstm_fixed.py
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

def create_sequences(data, sequence_length=50):
    """Creates sequences for LSTM training."""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

class SimpleLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(SimpleLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Let PyTorch handle hidden state initialization
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]  # Use the last time step
        predictions = self.linear(last_output)
        return predictions

def main():
    print("=== LSTM Training Started ===")
    
    # 1. Load the data
    data_path = os.path.join(PROJECT_ROOT, 'data', 'data_with_baseline_anomalies.csv')
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    print(f"Data loaded successfully. Shape: {df.shape}")

    # 2. Prepare the data for LSTM
    print("\n=== Preparing Data for LSTM ===")
    
    # Use the raw CPU values
    data = df['cpu_usage'].values
    print(f"Original CPU data shape: {data.shape}")
    
    # Split into train and test (time-based split)
    split_idx = int(len(data) * 0.8)  # 80% for training
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Scale the data
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
    test_data_scaled = scaler.transform(test_data.reshape(-1, 1)).flatten()
    
    # Create sequences
    sequence_length = 60
    X_train, y_train = create_sequences(train_data_scaled, sequence_length)
    X_test, y_test = create_sequences(test_data_scaled, sequence_length)
    
    print(f"X_train shape before reshape: {X_train.shape}")
    print(f"X_test shape before reshape: {X_test.shape}")
    
    # CRITICAL: Reshape to (samples, sequence_length, features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Convert to float32
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    print(f"Final X_train shape: {X_train.shape}")
    print(f"Final y_train shape: {y_train.shape}")
    print(f"Final X_test shape: {X_test.shape}")
    print(f"Final y_test shape: {y_test.shape}")
    
    # 3. Build the model
    print("\n=== Building LSTM Model ===")
    model = SimpleLSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1)
    print("Model built successfully!")
    
    # 4. Prepare data for training
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    
    print(f"Training tensor shapes - X: {X_train_tensor.shape}, y: {y_train_tensor.shape}")
    
    # Split for validation
    val_split = 0.1
    split_idx = int(len(X_train_tensor) * (1 - val_split))
    
    X_train_split = X_train_tensor[:split_idx]
    y_train_split = y_train_tensor[:split_idx]
    X_val_split = X_train_tensor[split_idx:]
    y_val_split = y_train_tensor[split_idx:]
    
    train_dataset = TensorDataset(X_train_split, y_train_split)
    val_dataset = TensorDataset(X_val_split, y_val_split)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 5. Train the model
    print("\n=== Starting Training ===")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []
    
    epochs = 30
    patience = 7
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                model.load_state_dict(best_model_state)
                break
    
    # 6. Save the model
    print("\n=== Saving Model ===")
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_save_path = os.path.join(models_dir, 'lstm_model.pth')
    torch.save(model.state_dict(), model_save_path)
    
    # Save the scaler
    import joblib
    scaler_save_path = os.path.join(models_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_save_path)
    
    print(f"Model saved to: {model_save_path}")
    print(f"Scaler saved to: {scaler_save_path}")
    
    # 7. Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    # Make predictions on test set
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        test_predictions = model(X_test_tensor).numpy()
    
    # Plot some test predictions
    plt.plot(y_test[:100], label='Actual', alpha=0.7)
    plt.plot(test_predictions[:100], label='Predicted', alpha=0.7)
    plt.title('Test Predictions vs Actual')
    plt.xlabel('Time Step')
    plt.ylabel('CPU Usage (scaled)')
    plt.legend()
    
    plt.tight_layout()
    
    reports_dir = os.path.join(PROJECT_ROOT, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    plt.savefig(os.path.join(reports_dir, 'lstm_results.png'))
    plt.show()
    
    print("\n=== Training Completed Successfully! ===")

if __name__ == "__main__":
    main()