import pandas as pd
import matplotlib.pyplot as plt

# Load the new data with predictions
df = pd.read_csv('data_with_baseline_anomalies.csv', parse_dates=['timestamp'])
df = df.set_index('timestamp').sort_index() # Ensure it's sorted by time

# Create a plot 
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

#CPU Usage with Anomalies
axes[0].plot(df.index, df['cpu_usage'], label='CPU Usage', color='blue', alpha=0.6, linewidth=1)
anomalies_cpu = df[df['baseline_anomaly'] == -1]
axes[0].scatter(anomalies_cpu.index, anomalies_cpu['cpu_usage'], color='red', label='Predicted Anomaly', alpha=0.7, s=15)
axes[0].set_title('CPU Usage with Baseline Anomaly Detection (Isolation Forest)')
axes[0].set_ylabel('CPU Usage %')
axes[0].legend()
axes[0].tick_params(axis='x', rotation=45)

# Plot 2: The Anomaly Score Itself
axes[1].plot(df.index, df['baseline_anomaly_score'], color='green', label='Anomaly Score', alpha=0.6)
axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5) # Add a line at zero
axes[1].scatter(anomalies_cpu.index, anomalies_cpu['baseline_anomaly_score'], color='red', alpha=0.7, s=15)
axes[1].set_title('Isolation Forest Anomaly Score')
axes[1].set_ylabel('Anomaly Score (lower = more anomalous)')
axes[1].set_xlabel('Timestamp')
axes[1].legend()
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


print("Top 5 most anomalous points (lowest score):")
print(df.nsmallest(5, 'baseline_anomaly_score')[['cpu_usage', 'memory_usage', 'baseline_anomaly_score']])