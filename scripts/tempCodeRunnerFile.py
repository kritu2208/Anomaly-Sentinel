from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import torch
import joblib
import asyncio
import aiohttp
import time
from datetime import datetime
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Anomaly Sentinel API", version="1.0.0")

class DockerAnomalyDetector:
    def __init__(self):
        self.models_loaded = False
        self.prometheus_url = os.getenv('PROMETHEUS_URL', 'http://localhost:9090')
        self.load_models()
        
    def load_models(self):
        """Load trained models"""
        try:
            # Load LSTM model
            model_path = "models/lstm_model.pth"
            scaler_path = "models/scaler.pkl"
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = torch.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.models_loaded = True
                logger.info("✅ Models loaded successfully")
            else:
                logger.warning("⚠️  Model files not found, running in demo mode")
                self.models_loaded = False
                
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            self.models_loaded = False
    
    async def get_prometheus_metrics(self):
        """Fetch metrics from Prometheus"""
        try:
            async with aiohttp.ClientSession() as session:
                # Query CPU usage
                cpu_query = '100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[1m])) * 100)'
                url = f"{self.prometheus_url}/api/v1/query"
                params = {'query': cpu_query}
                
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    
                    if data['status'] == 'success' and data['data']['result']:
                        cpu_usage = float(data['data']['result'][0]['value'][1])
                        return {
                            'timestamp': datetime.now(),
                            'cpu_usage': cpu_usage,
                            'success': True
                        }
                    
            return {'success': False, 'error': 'No data available'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def detect_anomaly(self, metrics_sequence):
        """Detect anomaly using LSTM model"""
        if not self.models_loaded or len(metrics_sequence) < 60:
            return {'anomaly': False, 'confidence': 0.5, 'reason': 'Insufficient data or models not loaded'}
        
        try:
            # Prepare sequence for LSTM
            sequence = np.array(metrics_sequence[-60:])  # Last 60 points
            sequence_scaled = self.scaler.transform(sequence.reshape(-1, 1)).flatten()
            
            # Convert to tensor
            sequence_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).unsqueeze(-1)
            
            # Make prediction
            with torch.no_grad():
                prediction = self.model(sequence_tensor)
                predicted_value = prediction.item()
            
            # Calculate error
            actual_value = sequence_scaled[-1]
            error = abs(actual_value - predicted_value)
            
            # Simple threshold (adjust based on your training)
            threshold = 1.5
            is_anomaly = error > threshold
            
            return {
                'anomaly': is_anomaly,
                'confidence': min(error / threshold, 1.0),
                'error': error,
                'threshold': threshold,
                'actual': float(self.scaler.inverse_transform([[actual_value]])[0][0]),
                'predicted': float(self.scaler.inverse_transform([[predicted_value]])[0][0])
            }
            
        except Exception as e:
            return {'anomaly': False, 'confidence': 0.5, 'reason': f'Model error: {str(e)}'}

# Initialize detector
detector = DockerAnomalyDetector()
metrics_history = []

@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    asyncio.create_task(metrics_collector())

async def metrics_collector():
    """Background task to collect metrics periodically"""
    while True:
        try:
            metrics = await detector.get_prometheus_metrics()
            if metrics['success']:
                metrics_history.append(metrics)
                # Keep only last 1000 points
                if len(metrics_history) > 1000:
                    metrics_history.pop(0)
                
                logger.info(f"📊 Collected metrics: {metrics['cpu_usage']:.2f}% CPU")
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
        
        await asyncio.sleep(15)  # Collect every 15 seconds

@app.get("/")
async def root():
    return {"message": "Anomaly Sentinel API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/metrics")
async def get_metrics():
    """Prometheus-compatible metrics endpoint"""
    if metrics_history:
        latest = metrics_history[-1]
        cpu_values = [m['cpu_usage'] for m in metrics_history[-60:]]  # Last 60 points
        anomaly_result = detector.detect_anomaly(cpu_values)
        
        return {
            "latest_metrics": latest,
            "anomaly_detection": anomaly_result,
            "history_count": len(metrics_history)
        }
    return {"error": "No metrics data available"}

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Web dashboard"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Anomaly Sentinel - Docker</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f0f2f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .card { background: white; padding: 20px; margin: 20px 0; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .status { padding: 15px; border-radius: 5px; margin: 10px 0; }
            .normal { background: #d4edda; border-left: 4px solid #28a745; }
            .warning { background: #fff3cd; border-left: 4px solid #ffc107; }
            .critical { background: #f8d7da; border-left: 4px solid #dc3545; }
            .chart-container { height: 300px; margin: 20px 0; }
            .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🐳 Anomaly Sentinel - Docker Deployment</h1>
            
            <div class="grid">
                <div class="card">
                    <h2>🚀 System Status</h2>
                    <div id="status" class="status normal">
                        <h3>✅ System Normal</h3>
                        <p>Monitoring active with Docker containers</p>
                    </div>
                    <p><strong>Models Loaded:</strong> <span id="modelsStatus">Checking...</span></p>
                    <p><strong>Data Points:</strong> <span id="dataPoints">0</span></p>
                    <p><strong>Last Update:</strong> <span id="lastUpdate">-</span></p>
                </div>
                
                <div class="card">
                    <h2>📊 Service Status</h2>
                    <div id="services">
                        <p>✅ Prometheus: <span id="promStatus">Checking...</span></p>
                        <p>✅ Grafana: <span id="grafanaStatus">Checking...</span></p>
                        <p>✅ Node Exporter: <span id="nodeStatus">Checking...</span></p>
                        <p>✅ Anomaly API: <span id="apiStatus">Running</span></p>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>📈 Real-time Monitoring</h2>
                <div class="chart-container">
                    <canvas id="metricsChart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h2>🚨 Anomaly Detection</h2>
                <div id="anomalyInfo">
                    <p>Waiting for data...</p>
                </div>
            </div>
        </div>

        <script>
            let metricsChart;
            
            async function updateDashboard() {
                try {
                    const response = await fetch('/metrics');
                    const data = await response.json();
                    
                    if (data.latest_metrics) {
                        // Update status
                        document.getElementById('dataPoints').textContent = data.history_count;
                        document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
                        document.getElementById('modelsStatus').textContent = data.anomaly_detection ? 'Loaded' : 'Demo Mode';
                        
                        // Update chart
                        updateChart(data);
                        
                        // Update anomaly info
                        updateAnomalyInfo(data.anomaly_detection);
                    }
                } catch (error) {
                    console.error('Error updating dashboard:', error);
                }
            }
            
            function updateChart(data) {
                // This would need historical data - for now show current state
                const ctx = document.getElementById('metricsChart').getContext('2d');
                
                if (!metricsChart) {
                    metricsChart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: ['Current'],
                            datasets: [{
                                label: 'CPU Usage %',
                                data: [data.latest_metrics.cpu_usage],
                                borderColor: 'rgb(75, 192, 192)',
                                tension: 0.1
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                y: { beginAtZero: true, max: 100 }
                            }
                        }
                    });
                } else {
                    metricsChart.data.datasets[0].data = [data.latest_metrics.cpu_usage];
                    metricsChart.update();
                }
            }
            
            function updateAnomalyInfo(anomalyData) {
                const container = document.getElementById('anomalyInfo');
                const statusDiv = document.getElementById('status');
                
                if (anomalyData.anomaly) {
                    statusDiv.className = 'status critical';
                    statusDiv.innerHTML = '<h3>🚨 CRITICAL ANOMALY DETECTED</h3><p>Immediate attention required</p>';
                    
                    container.innerHTML = `
                        <div style="color: #dc3545;">
                            <p><strong>Anomaly Score:</strong> ${anomalyData.error.toFixed(3)}</p>
                            <p><strong>Confidence:</strong> ${(anomalyData.confidence * 100).toFixed(1)}%</p>
                            <p><strong>Actual CPU:</strong> ${anomalyData.actual.toFixed(2)}%</p>
                            <p><strong>Predicted CPU:</strong> ${anomalyData.predicted.toFixed(2)}%</p>
                        </div>
                    `;
                } else {
                    statusDiv.className = 'status normal';
                    statusDiv.innerHTML = '<h3>✅ SYSTEM NORMAL</h3><p>All metrics within expected ranges</p>';
                    
                    container.innerHTML = `
                        <p><strong>Status:</strong> Normal</p>
                        <p><strong>Current Score:</strong> ${anomalyData.error ? anomalyData.error.toFixed(3) : 'N/A'}</p>
                        <p><strong>Threshold:</strong> ${anomalyData.threshold || 'N/A'}</p>
                    `;
                }
            }
            
            // Update every 5 seconds
            setInterval(updateDashboard, 5000);
            updateDashboard();
        </script>
    </body>
    </html>
    """
    return html

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Anomaly Sentinel API in Docker mode...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")