import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Add project root to Python path

import pandas as pd
from src.feat_engg import engineer_features
from src.base_model import train_baseline_model

def main():
    # 1. Load our generated data
    data_path = "../data/sample_metrics.csv"
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # 2. Perform feature engineering
    df_engineered = engineer_features(df)

    # 3. Train the baseline model
    df_with_anomalies, model, scaler, features_used = train_baseline_model(df_engineered)

    # 4. Save the results for visualization
    output_path = "../data/data_with_baseline_anomalies.csv"
    df_with_anomalies.to_csv(output_path)
    print(f"Data with baseline predictions saved to {output_path}")
    print(f"Features used: {features_used}")

    # 5. Let's see some stats
    num_anomalies = (df_with_anomalies['baseline_anomaly'] == -1).sum()
    total_points = len(df_with_anomalies)
    print(f"Baseline model detected {num_anomalies} anomalies ({num_anomalies/total_points*100:.2f}% of data).")

if __name__ == "__main__":
    main()