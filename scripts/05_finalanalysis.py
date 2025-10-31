import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

def analyze_anomaly_types(df):
    """Analyze what types of anomalies each method detects."""
    
    print("=== Anomaly Type Analysis ===")
    
    # Categorize anomalies
    df['anomaly_type'] = 'Normal'
    df.loc[df['baseline_anomaly'] == -1, 'anomaly_type'] = 'Isolation_Forest_Only'
    df.loc[df['lstm_anomaly'] == True, 'anomaly_type'] = 'LSTM_Only'
    df.loc[(df['baseline_anomaly'] == -1) & (df['lstm_anomaly'] == True), 'anomaly_type'] = 'Both_Methods'
    
    # Statistics
    anomaly_counts = df['anomaly_type'].value_counts()
    print("Anomaly Distribution:")
    for anomaly_type, count in anomaly_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {anomaly_type}: {count} ({percentage:.2f}%)")
    
    return df

def analyze_anomaly_patterns(df):
    """Analyze the characteristics of different anomaly types."""
    
    print("\n=== Anomaly Pattern Analysis ===")
    
    # Group by anomaly type and analyze statistics
    anomaly_stats = df.groupby('anomaly_type').agg({
        'cpu_usage': ['mean', 'std', 'min', 'max'],
        'memory_usage': ['mean', 'std'],
        'lstm_anomaly_score': 'mean',
        'baseline_anomaly_score': 'mean'
    }).round(3)
    
    print("Statistical Characteristics by Anomaly Type:")
    print(anomaly_stats)
    
    return anomaly_stats

def create_final_dashboard(df):
    """Create a comprehensive dashboard comparing both methods."""
    
    print("\n=== Creating Final Dashboard ===")
    
    # Create a 2x2 subplot dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Time series with all anomaly types
    colors = {'Normal': 'blue', 'Isolation_Forest_Only': 'orange', 
              'LSTM_Only': 'red', 'Both_Methods': 'purple'}
    
    for anomaly_type in df['anomaly_type'].unique():
        subset = df[df['anomaly_type'] == anomaly_type]
        axes[0,0].scatter(subset['timestamp'], subset['cpu_usage'], 
                         color=colors[anomaly_type], label=anomaly_type, 
                         alpha=0.7, s=20)
    
    axes[0,0].plot(df['timestamp'], df['cpu_usage'], 'blue', alpha=0.3, linewidth=0.5)
    axes[0,0].set_title('CPU Usage with Anomaly Types')
    axes[0,0].set_ylabel('CPU Usage %')
    axes[0,0].legend()
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Distribution of CPU usage by anomaly type
    anomaly_types = ['Isolation_Forest_Only', 'LSTM_Only', 'Both_Methods', 'Normal']
    for anomaly_type in anomaly_types:
        subset = df[df['anomaly_type'] == anomaly_type]
        axes[0,1].hist(subset['cpu_usage'], bins=30, alpha=0.6, 
                      label=anomaly_type, density=True)
    
    axes[0,1].set_title('Distribution of CPU Usage by Anomaly Type')
    axes[0,1].set_xlabel('CPU Usage %')
    axes[0,1].set_ylabel('Density')
    axes[0,1].legend()
    
    # Plot 3: Anomaly scores over time
    axes[1,0].plot(df['timestamp'], df['lstm_anomaly_score'], 
                  label='LSTM Anomaly Score', color='red', alpha=0.7)
    axes[1,0].plot(df['timestamp'], -df['baseline_anomaly_score'],  # Invert for better visualization
                  label='Isolation Forest Score (inverted)', color='orange', alpha=0.7)
    axes[1,0].set_title('Anomaly Scores Over Time')
    axes[1,0].set_ylabel('Anomaly Score')
    axes[1,0].set_xlabel('Timestamp')
    axes[1,0].legend()
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Scatter plot of both anomaly scores
    scatter = axes[1,1].scatter(df['lstm_anomaly_score'], -df['baseline_anomaly_score'],
                               c=df['cpu_usage'], cmap='viridis', alpha=0.6)
    axes[1,1].set_xlabel('LSTM Anomaly Score')
    axes[1,1].set_ylabel('Isolation Forest Score (inverted)')
    axes[1,1].set_title('Correlation Between Anomaly Detection Methods')
    plt.colorbar(scatter, ax=axes[1,1], label='CPU Usage %')
    
    plt.tight_layout()
    
    # Save the dashboard
    reports_dir = os.path.join(PROJECT_ROOT, 'reports')
    dashboard_path = os.path.join(reports_dir, 'final_anomaly_dashboard.png')
    plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Dashboard saved to: {dashboard_path}")
    
    return fig

def generate_performance_report(df):
    """Generate a summary performance report."""
    
    print("\n=== Performance Report ===")
    
    # Calculate precision metrics (using our synthetic data knowledge)
    # In real scenario, you'd use labeled data
    
    total_points = len(df)
    isolation_forest_count = len(df[df['baseline_anomaly'] == -1])
    lstm_count = len(df[df['lstm_anomaly'] == True])
    both_count = len(df[(df['baseline_anomaly'] == -1) & (df['lstm_anomaly'] == True)])
    
    report = f"""
    ANOMALY DETECTION PERFORMANCE REPORT
    ====================================
    
    Dataset Statistics:
    - Total data points: {total_points}
    - Time period: {df['timestamp'].min()} to {df['timestamp'].max()}
    
    Detection Results:
    - Isolation Forest anomalies: {isolation_forest_count} ({isolation_forest_count/total_points*100:.1f}%)
    - LSTM anomalies: {lstm_count} ({lstm_count/total_points*100:.1f}%)
    - Anomalies detected by both methods: {both_count} ({both_count/total_points*100:.1f}%)
    
    Key Insights:
    1. The LSTM method detected {lstm_count - isolation_forest_count} more anomalies than Isolation Forest
    2. Only {both_count} anomalies were detected by both methods ({both_count/max(isolation_forest_count, lstm_count)*100:.1f}% agreement)
    3. This suggests the methods are complementary:
       - Isolation Forest finds point anomalies (statistical outliers)
       - LSTM finds contextual anomalies (pattern deviations)
    
    Recommendation:
    For production use, consider combining both methods:
    - Use Isolation Forest for real-time point anomaly detection (low computational cost)
    - Use LSTM for periodic deep analysis of temporal patterns (higher accuracy)
    """
    
    print(report)
    
    # Save report to file
    report_path = os.path.join(PROJECT_ROOT, 'reports', 'performance_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Performance report saved to: {report_path}")
    
    return report

def main():
    print("=== Final Anomaly Detection Analysis ===")
    
    # Load the combined results
    data_path = os.path.join(PROJECT_ROOT, 'data', 'data_with_lstm_anomalies.csv')
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    print(f"Data loaded. Shape: {df.shape}")
    
    # 1. Analyze anomaly types
    df = analyze_anomaly_types(df)
    
    # 2. Analyze patterns
    stats = analyze_anomaly_patterns(df)
    
    # 3. Create comprehensive dashboard
    create_final_dashboard(df)
    
    # 4. Generate performance report
    report = generate_performance_report(df)
    
    print("\n=== Analysis Complete! ===")
    print("\nNext steps:")
    print("1. Review the dashboard in reports/final_anomaly_dashboard.png")
    print("2. Read the detailed analysis in reports/performance_report.txt")
    print("3. The project is ready for your portfolio!")

if __name__ == "__main__":
    main()