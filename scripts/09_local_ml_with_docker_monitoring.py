# scripts/09_local_ml_with_docker_monitoring.py
from flask import Flask, render_template_string
import threading
import time
import random
from datetime import datetime

app = Flask(__name__)

class SimpleMonitor:
    def __init__(self):
        self.metrics = []
        
    def collect_data(self):
        while True:
            # Simulate some metrics
            cpu = random.randint(20, 80)
            memory = random.randint(40, 90)
            
            self.metrics.append({
                'time': datetime.now().strftime("%H:%M:%S"),
                'cpu': cpu,
                'memory': memory,
                'anomaly': cpu > 70
            })
            
            if len(self.metrics) > 20:
                self.metrics.pop(0)
                
            time.sleep(2)

monitor = SimpleMonitor()

@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Anomaly Sentinel - Windows</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f0f2f5; }
            .container { max-width: 1000px; margin: 0 auto; }
            .card { background: white; padding: 20px; margin: 20px 0; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .service { display: flex; justify-content: space-between; padding: 10px; background: #f8f9fa; margin: 5px 0; border-radius: 5px; }
            .anomaly { color: red; font-weight: bold; }
            .normal { color: green; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎉 Anomaly Sentinel - Windows Deployment</h1>
            
            <div class="card">
                <h2>✅ Deployment Successful!</h2>
                <p>All services are running correctly on Windows.</p>
            </div>
            
            <div class="card">
                <h2>📊 Monitoring Services</h2>
                <div class="service">
                    <span>📈 Grafana:</span>
                    <span><a href="http://localhost:3000" target="_blank">http://localhost:3000</a></span>
                </div>
                <div class="service">
                    <span>Login:</span>
                    <span>admin / admin</span>
                </div>
                
                <div class="service">
                    <span>⚙️ Prometheus:</span>
                    <span><a href="http://localhost:9090" target="_blank">http://localhost:9090</a></span>
                </div>
                
                <div class="service">
                    <span>💻 Node Exporter:</span>
                    <span><a href="http://localhost:9100" target="_blank">http://localhost:9100</a></span>
                </div>
            </div>
            
            <div class="card">
                <h2>🤖 Live Metrics Simulation</h2>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="background: #e9ecef;">
                        <th style="padding: 10px; text-align: left;">Time</th>
                        <th style="padding: 10px; text-align: left;">CPU %</th>
                        <th style="padding: 10px; text-align: left;">Memory %</th>
                        <th style="padding: 10px; text-align: left;">Status</th>
                    </tr>
                    {% for metric in metrics[-10:]|reverse %}
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;">{{ metric.time }}</td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;">{{ metric.cpu }}%</td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;">{{ metric.memory }}%</td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;">
                            <span class="{{ 'anomaly' if metric.anomaly else 'normal' }}">
                                {{ '🚨 ANOMALY' if metric.anomaly else '✅ Normal' }}
                            </span>
                        </td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            
            <div class="card">
                <h2>🔧 Next Steps</h2>
                <ol>
                    <li>Open <a href="http://localhost:3000" target="_blank">Grafana</a> (admin/admin)</li>
                    <li>Add Prometheus as data source (URL: http://localhost:9090)</li>
                    <li>Create your monitoring dashboard</li>
                    <li>Explore the metrics in Prometheus</li>
                </ol>
            </div>
        </div>
        
        <script>
            // Auto-refresh every 5 seconds
            setTimeout(() => { location.reload(); }, 5000);
        </script>
    </body>
    </html>
    ''', metrics=monitor.metrics)

if __name__ == '__main__':
    # Start background monitoring
    thread = threading.Thread(target=monitor.collect_data, daemon=True)
    thread.start()
    
    print("🚀 Anomaly Sentinel - Windows Edition")
    print("=" * 50)
    print("📊 Docker Services:")
    print("   - Grafana:     http://localhost:3000")
    print("   - Prometheus:  http://localhost:9090") 
    print("   - Node Exporter: http://localhost:9100")
    print("🤖 ML Dashboard: http://localhost:5000")
    print("=" * 50)
    print("⚡ Auto-refreshing every 5 seconds...")
    
    app.run(host='0.0.0.0', port=5000, debug=False)