import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

class SimpleLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(SimpleLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        predictions = self.linear(last_output)
        return predictions

def create_sequences(data, sequence_length=50):
    """Creates sequences for LSTM prediction."""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def detect_anomalies_with_lstm(model, data, threshold_std=2.0):
    """
    Detect anomalies using LSTM prediction errors.
    Returns anomaly scores and indices.
    """
    model.eval()
    with torch.no_grad():
        data_tensor = torch.FloatTensor(data)
        predictions = model(data_tensor).numpy().flatten()
    
    # Calculate prediction errors (anomaly scores)
    actual_values = data[:, -1, 0]  # Last value in each sequence
    prediction_errors = np.abs(actual_values - predictions)
    
    # Calculate anomaly threshold (based on standard deviation)
    error_mean = np.mean(prediction_errors)
    error_std = np.std(prediction_errors)
    threshold = error_mean + threshold_std * error_std
    
    # Identify anomalies
    anomalies = prediction_errors > threshold
    anomaly_scores = prediction_errors
    
    return anomalies, anomaly_scores, threshold, predictions

def main():
    print("=== LSTM Anomaly Detection Started ===")
    
    # 1. Load the data
    data_path = os.path.join(PROJECT_ROOT, 'data', 'data_with_baseline_anomalies.csv')
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    print(f"Data loaded. Shape: {df.shape}")
    
    # 2. Load the trained model and scaler
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    model_path = os.path.join(models_dir, 'lstm_model.pth')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    
    print("Loading trained model and scaler...")
    model = SimpleLSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1)
    model.load_state_dict(torch.load(model_path))
    scaler = joblib.load(scaler_path)
    
    # 3. Prepare the data for anomaly detection
    data = df['cpu_usage'].values
    data_scaled = scaler.transform(data.reshape(-1, 1)).flatten()
    
    # Create sequences (use the entire dataset)
    sequence_length = 60
    X_sequences, y_actual = create_sequences(data_scaled, sequence_length)
    X_sequences = X_sequences.reshape(X_sequences.shape[0], X_sequences.shape[1], 1)
    X_sequences = X_sequences.astype(np.float32)
    
    print(f"Sequences for anomaly detection: {X_sequences.shape}")
    
    # 4. Detect anomalies
    print("Detecting anomalies with LSTM...")
    anomalies, anomaly_scores, threshold, predictions = detect_anomalies_with_lstm(
        model, X_sequences, threshold_std=2.0
    )
    
    num_anomalies = np.sum(anomalies)
    print(f"LSTM detected {num_anomalies} anomalies ({num_anomalies/len(anomalies)*100:.2f}% of data)")
    print(f"Anomaly threshold: {threshold:.4f}")
    
    # 5. Add results to dataframe
    # Align the anomalies with the original timestamps (account for sequence offset)
    result_df = df.iloc[sequence_length:].copy()  # Skip the first 'sequence_length' points
    result_df['lstm_anomaly_score'] = anomaly_scores
    result_df['lstm_anomaly'] = anomalies
    result_df['lstm_prediction'] = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    result_df['lstm_prediction_error'] = np.abs(result_df['cpu_usage'] - result_df['lstm_prediction'])
    
    # 6. Save results
    output_path = os.path.join(PROJECT_ROOT, 'data', 'data_with_lstm_anomalies.csv')
    result_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    
    # 7. Create comparison visualization
    print("Creating visualization...")
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Original data with both anomaly detection methods
    axes[0].plot(result_df['timestamp'], result_df['cpu_usage'], 
                label='CPU Usage', color='blue', alpha=0.7, linewidth=1)
    
    # Plot Isolation Forest anomalies (from baseline)
    baseline_anomalies = result_df[result_df['baseline_anomaly'] == -1]
    axes[0].scatter(baseline_anomalies['timestamp'], baseline_anomalies['cpu_usage'],
                   color='orange', label='Isolation Forest Anomalies', alpha=0.7, s=20)
    
    # Plot LSTM anomalies
    lstm_anomalies = result_df[result_df['lstm_anomaly'] == True]
    axes[0].scatter(lstm_anomalies['timestamp'], lstm_anomalies['cpu_usage'],
                   color='red', label='LSTM Anomalies', alpha=0.8, s=30, marker='x')
    
    axes[0].set_title('CPU Usage with Anomaly Detection Comparison')
    axes[0].set_ylabel('CPU Usage %')
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot 2: LSTM Prediction vs Actual
    axes[1].plot(result_df['timestamp'], result_df['cpu_usage'], 
                label='Actual CPU Usage', color='blue', alpha=0.7)
    axes[1].plot(result_df['timestamp'], result_df['lstm_prediction'],
                label='LSTM Prediction', color='green', alpha=0.7)
    axes[1].set_title('LSTM Predictions vs Actual Values')
    axes[1].set_ylabel('CPU Usage %')
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Anomaly Scores
    axes[2].plot(result_df['timestamp'], result_df['lstm_anomaly_score'],
                label='LSTM Anomaly Score', color='purple', alpha=0.7)
    axes[2].axhline(y=threshold, color='red', linestyle='--', 
                   label=f'Anomaly Threshold ({threshold:.3f})')
    axes[2].set_title('LSTM Anomaly Scores')
    axes[2].set_ylabel('Anomaly Score (Prediction Error)')
    axes[2].set_xlabel('Timestamp')
    axes[2].legend()
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    reports_dir = os.path.join(PROJECT_ROOT, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    plot_path = os.path.join(reports_dir, 'anomaly_detection_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to: {plot_path}")
    
    # 8. Compare the two methods
    print("\n=== Method Comparison ===")
    baseline_anomaly_count = len(baseline_anomalies)
    lstm_anomaly_count = len(lstm_anomalies)
    
    # Find common anomalies detected by both methods
    common_anomalies = result_df[
        (result_df['baseline_anomaly'] == -1) & 
        (result_df['lstm_anomaly'] == True)
    ]
    
    print(f"Isolation Forest anomalies: {baseline_anomaly_count}")
    print(f"LSTM anomalies: {lstm_anomaly_count}")
    print(f"Common anomalies (both methods): {len(common_anomalies)}")
    print(f"Agreement between methods: {len(common_anomalies)/max(baseline_anomaly_count, lstm_anomaly_count)*100:.1f}%")
    
    print("\n=== LSTM Anomaly Detection Completed! ===")

if __name__ == "__main__":
    main()