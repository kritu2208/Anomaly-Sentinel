# scripts/generate_sample_data.py
import pandas as pd
import numpy as np
import time
import os  # <-- Import the os module to handle directory creation

def generate_metrics(timesteps=10000):
    """Generates synthetic CPU, memory, and disk I/O metrics."""
    timestamps = pd.date_range(start='2024-01-01', periods=timesteps, freq='s') # Use lowercase 's' instead of 'S'
    cpu_usage = np.random.normal(50, 10, timesteps) # Base signal

    # Inject some anomalies
    # 1. Short Spikes (e.g., brief process launch)
    spike_indices = np.random.choice(timesteps, size=50, replace=False)
    cpu_usage[spike_indices] += np.random.uniform(30, 80, 50)

    # 2. Ramps (e.g., memory leak, growing queue)
    ramp_start = np.random.choice(timesteps - 500, size=3, replace=False)
    for start in ramp_start:
        ramp_length = np.random.randint(100, 500)
        cpu_usage[start:start+ramp_length] += np.linspace(0, 70, ramp_length)

    # 3. Level Shifts (e.g., background job starts)
    shift_points = np.random.choice(timesteps - 1000, size=2, replace=False)
    for point in shift_points:
        cpu_usage[point:] += np.random.uniform(10, 25)

    # Clip values to be between 0 and 100
    cpu_usage = np.clip(cpu_usage, 0, 100)

    # Simulate correlated memory usage (often correlates with CPU)
    memory_usage = 0.7 * cpu_usage + np.random.normal(30, 5, timesteps)
    memory_usage = np.clip(memory_usage, 0, 100)

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage,
        # Add disk_io later if you want
    })
    return df

if __name__ == "__main__":
    print("Generating sample data...")
    df = generate_metrics()
    
    # Define the path to the data directory
    data_dir = '../data'
    # Check if the directory exists, and if not, create it.
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
    
    # Save to data/ directory
    filepath = os.path.join(data_dir, 'sample_metrics.csv')
    df.to_csv(filepath, index=False)
    print(f"Data generated and saved to {filepath}")
    print(df.head())