# src/lstm_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
    # Initialize hidden state with zeros - FIXED DIMENSIONS
    # For batch_first=True: (num_layers, batch_size, hidden_size)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        # Forward propagate LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.dropout(out[:, -1, :])  # Take the last output
        out = self.linear(out)
        return out

def build_lstm_model(sequence_length=50):
    """Builds and returns the LSTM model"""
    model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1)
    return model

def train_lstm_model(model, X_train, y_train, epochs=20, learning_rate=0.001, validation_split=0.1, patience=5):
    """Trains the LSTM model with early stopping"""
    # Convert numpy arrays to PyTorch tensors - FIXED: Add batch dimension
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    
    print(f"X_train shape: {X_train_tensor.shape}")
    print(f"y_train shape: {y_train_tensor.shape}")
    
    # Split into training and validation
    split_idx = int(len(X_train_tensor) * (1 - validation_split))
    X_train_split = X_train_tensor[:split_idx]
    y_train_split = y_train_tensor[:split_idx]
    X_val_split = X_train_tensor[split_idx:]
    y_val_split = y_train_tensor[split_idx:]
    
    print(f"Training split shape: {X_train_split.shape}")
    print(f"Validation split shape: {X_val_split.shape}")
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_split, y_train_split)
    val_dataset = TensorDataset(X_val_split, y_val_split)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
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
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                # Load the best model
                model.load_state_dict(torch.load('best_model.pth'))
                break
    
    # Clean up temporary file
    if os.path.exists('best_model.pth'):
        os.remove('best_model.pth')
    
    history = {'train_loss': train_losses, 'val_loss': val_losses}
    return history, model