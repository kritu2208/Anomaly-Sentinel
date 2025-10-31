# src/feature_engineering.py
import pandas as pd
import numpy as np

def create_rolling_features(df, column, window_sizes=[60, 300]):
    """
    Creates rolling statistical features for a given column.
    window_sizes: list of window sizes in seconds (since our data is 1-second freq)
    """
    df = df.copy()
    for window in window_sizes:
        # Rolling Average (Trend)
        df[f'{column}_rolling_mean_{window}'] = df[column].rolling(window=window, min_periods=1).mean()
        # Rolling Standard Deviation (Volatility)
        df[f'{column}_rolling_std_{window}'] = df[column].rolling(window=window, min_periods=1).std()
        # Z-Score for this window (How many std devs away from recent mean)
        df[f'{column}_zscore_{window}'] = (df[column] - df[f'{column}_rolling_mean_{window}']) / df[f'{column}_rolling_std_{window}'].replace(0, 1e-10) # Avoid div by zero

    return df

# src/feature_engineering.py (Updated function)
def create_temporal_features(df):
    """Extracts temporal features from the timestamp INDEX."""
    df = df.copy()
    # Use the index, which is now a DatetimeIndex
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    # Cyclical encoding for hour (sin/cos)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    return df

# src/feature_engineering.py (Slight update to the main function)
def engineer_features(df):
    """Main function to run all feature engineering steps."""
    print("Engineering features...")
    # Ensure timestamp is datetime and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()

    # Create features for each metric
    # NOTE: We use the column names, but 'timestamp' is now the index, so it's safe.
    for column in ['cpu_usage', 'memory_usage']:
        if column in df.columns: # Safety check
            df = create_rolling_features(df, column)

    df = create_temporal_features(df) # This now uses the index

    # Handle NaN values created by rolling windows
    df = df.fillna(method='bfill')

    print("Feature engineering complete.")
    return df