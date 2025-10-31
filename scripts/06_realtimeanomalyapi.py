# scripts/06_realtime_anomaly_api.py
import asyncio
import pandas as pd
import numpy as np
from prometheus_api_client import PrometheusConnect
from datetime import datetime, timedelta
import torch
import joblib
import warnings
warnings.filterwarnings('ignore')

class RealTimeAnomalyDetector:
    def __init__(self, model_path, scaler_path):
        # Load trained model
        self.model = torch.load(model_path)
        self.model.eval()
        self.scaler = joblib.load(scaler_path)
        
        # Connect to Prometheus
        self.prom = PrometheusConnect(url="http://localhost:9090", disable_ssl=True)
        
        # Buffer for recent metrics
        self.metric_buffer = []
        self.buffer_size = 100
        self.sequence_length = 60
        
    async def collect_metrics(self):
        """Continuously collect metrics from Prometheus"""
        while True:
            try:
                # Query current CPU usage
                cpu_query = '100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[1m])) * 100)'
                result = self.prom.custom_query(query=cpu_query)
                
                if result:
                    cpu_usage = float(result[0]['value'][1])
                    timestamp = datetime.now()
                    
                    # Add to buffer
                    self.metric_buffer.append({
                        'timestamp': timestamp,
                        'cpu_usage': cpu_usage
                    })
                    
                    # Keep buffer size limited
                    if len(self.metric_buffer) > self.buffer_size:
                        self.metric_buffer.pop(0)
                    
                    print(f"📊 Collected metric: {cpu_usage:.2f}% CPU at {timestamp}")
                
            except Exception as e:
                print(f"❌ Error collecting metrics: {e}")
            
            await asyncio.sleep(15)  # Collect every 15 seconds
    
    def detect_anomaly(self):
        """Detect anomalies using the latest metrics"""
        if len(self.metric_buffer) < self.sequence_length:
            return None, "Insufficient data"
        
        # Prepare the most recent sequence
        recent_metrics = self.metric_buffer[-self.sequence_length:]
        cpu_values = [m['cpu_usage'] for m in recent_metrics]
        
        # Scale the data
        cpu_scaled = self.scaler.transform(np.array(cpu_values).reshape(-1, 1)).flatten()
        
        # Prepare for LSTM (sequence_length, 1)
        sequence = torch.FloatTensor(cpu_scaled).unsqueeze(0).unsqueeze(-1)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(sequence)
            predicted_value = prediction.item()
        
        # Calculate error (anomaly score)
        actual_value = cpu_scaled[-1]
        error = abs(actual_value - predicted_value)
        
        # Simple threshold (you can make this dynamic)
        threshold = 1.5
        is_anomaly = error > threshold
        
        return {
            'timestamp': datetime.now(),
            'actual_cpu': cpu_values[-1],
            'predicted_cpu': float(self.scaler.inverse_transform([[predicted_value]])[0][0]),
            'anomaly_score': error,
            'is_anomaly': is_anomaly,
            'threshold': threshold
        }
    
    async def monitor_loop(self):
        """Main monitoring loop"""
        print("🚀 Starting real-time anomaly detection...")
        
        # Start metric collection in background
        asyncio.create_task(self.collect_metrics())
        
        while True:
            try:
                if len(self.metric_buffer) >= self.sequence_length:
                    result = self.detect_anomaly()
                    
                    if result and result['is_anomaly']:
                        print(f"🚨 ANOMALY DETECTED!")
                        print(f"   Time: {result['timestamp']}")
                        print(f"   Actual CPU: {result['actual_cpu']:.2f}%")
                        print(f"   Predicted CPU: {result['predicted_cpu']:.2f}%")
                        print(f"   Anomaly Score: {result['anomaly_score']:.3f}")
                        print("-" * 50)
                    
                    elif result:
                        print(f"✅ Normal - Score: {result['anomaly_score']:.3f}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"❌ Error in monitoring loop: {e}")
                await asyncio.sleep(10)

# FastAPI version for web dashboard
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()
detector = None

@app.on_event("startup")
async def startup_event():
    global detector
    # Initialize detector (you'll need to modify model loading for the real-time case)
    print("Initializing anomaly detector...")

@app.get("/")
async def root():
    return {"message": "Anomaly Detection API Running"}

@app.get("/metrics")
async def get_metrics():
    if detector and detector.metric_buffer:
        latest = detector.metric_buffer[-1]
        anomaly_result = detector.detect_anomaly()
        return {
            "latest_metric": latest,
            "anomaly_status": anomaly_result
        }
    return {"error": "No data available"}

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Simple dashboard showing current status"""
    html_content = """
    <html>
        <head>
            <title>Anomaly Sentinel Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .normal { color: green; }
                .anomaly { color: red; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>🚀 Anomaly Sentinel - Live Dashboard</h1>
            <div id="status">Loading...</div>
            <script>
                async function updateStatus() {
                    const response = await fetch('/metrics');
                    const data = await response.json();
                    
                    if (data.anomaly_status) {
                        const status = data.anomaly_status;
                        const statusDiv = document.getElementById('status');
                        statusDiv.innerHTML = `
                            <h2 class="${status.is_anomaly ? 'anomaly' : 'normal'}">
                                ${status.is_anomaly ? '🚨 ANOMALY DETECTED' : '✅ System Normal'}
                            </h2>
                            <p>CPU Usage: ${status.actual_cpu ? status.actual_cpu.toFixed(2) : 'N/A'}%</p>
                            <p>Predicted: ${status.predicted_cpu ? status.predicted_cpu.toFixed(2) : 'N/A'}%</p>
                            <p>Anomaly Score: ${status.anomaly_score ? status.anomaly_score.toFixed(3) : 'N/A'}</p>
                            <p>Last Updated: ${new Date().toLocaleTimeString()}</p>
                        `;
                    }
                }
                setInterval(updateStatus, 5000);
                updateStatus();
            </script>
        </body>
    </html>
    """
    return html_content

async def main():
    """Run the real-time detector"""
    # Note: You'll need to adapt your model for real-time use
    # For now, let's run the API
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    # For now, run the simple version without model integration
    asyncio.run(main())