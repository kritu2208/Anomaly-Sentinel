# scripts/deploy_production.py
import os
import subprocess
import sys
import time

def run_command(command, description):
    """Run a shell command with error handling"""
    print(f"🚀 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print("🎯 Starting Production Deployment of AnomalySentinel")
    
    # 1. Start monitoring infrastructure
    print("\n1. Starting Prometheus + Grafana stack...")
    if not run_command("docker-compose -f docker-compose.yml up -d", "Start Prometheus/Grafana"):
        return
    
    if not run_command("docker-compose -f docker-compose.node.yml up -d", "Start Node Exporter"):
        return
    
    # 2. Wait for services to start
    print("\n2. Waiting for services to initialize...")
    time.sleep(30)
    
    # 3. Start the anomaly detection API
    print("\n3. Starting Anomaly Detection API...")
    api_process = subprocess.Popen([
        sys.executable, "scripts/06_realtime_anomaly_api.py"
    ])
    
    # 4. Provide user information
    print("\n🎉 Deployment Complete!")
    print("\n📊 Access your services:")
    print("   - Grafana Dashboard: http://localhost:3000 (admin/admin)")
    print("   - Prometheus: http://localhost:9090")
    print("   - Anomaly API: http://localhost:8000")
    print("   - Anomaly Dashboard: http://localhost:8000/dashboard")
    
    print("\n🔧 Next steps:")
    print("   1. Configure Grafana datasource to point to Prometheus")
    print("   2. Import the dashboard JSON provided in /grafana/dashboard.json")
    print("   3. Monitor your system in real-time!")
    
    try:
        api_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        api_process.terminate()

if __name__ == "__main__":
    main()