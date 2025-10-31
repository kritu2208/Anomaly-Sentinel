# src/baseline_model.py
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def train_baseline_model(df, contamination=0.03):
    """
    Trains an Isolation Forest model on the engineered features.
    Contamination is an estimate of the proportion of anomalies. We set it low.
    """
    # 1. Select features for the model (use our new rolling features)
    feature_columns = [col for col in df.columns if 'rolling' in col or 'zscore' in col]
    X = df[feature_columns]

    # 2. Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Train the Isolation Forest
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42
    )
    model.fit(X_scaled)

    # 4. Predict anomalies (-1 for anomaly, 1 for normal)
    predictions = model.predict(X_scaled)
    df['baseline_anomaly'] = predictions
    df['baseline_anomaly_score'] = model.decision_function(X_scaled) # The lower the score, the more anomalous

    print("Baseline Isolation Forest model trained.")
    return df, model, scaler, feature_columns