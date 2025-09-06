#!/usr/bin/env python3
"""
AML 360º Setup and Quick Start
"""

import os
import subprocess
import sys
from pathlib import Path

def check_python_version():
    """Check Python version"""
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_requirements():
    """Install Python requirements"""
    print("📦 Installing requirements...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def setup_directories():
    """Create necessary directories"""
    dirs = [
        "data/raw", "data/processed", "data/external",
        "models", "logs", "outputs", "artifacts"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        
    print(f"✅ Created {len(dirs)} directories")

def create_env_file():
    """Create .env template file"""
    env_content = """# AML 360º Environment Variables

# BigQuery Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
BIGQUERY_DATASET=aml
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Redis Configuration  
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# API Configuration
API_KEY_HASH=your-sha256-hash-here
ENABLE_CACHING=true
CACHE_TTL=3600

# Model Configuration
MODEL_PATH=/models/aml_ensemble.pkl

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
"""
    
    env_path = Path(".env")
    if not env_path.exists():
        with open(env_path, "w") as f:
            f.write(env_content)
        print("✅ Created .env template file")
    else:
        print("ℹ️  .env file already exists")

def quick_test():
    """Run quick system test"""
    print("🧪 Running quick system test...")
    
    try:
        # Test imports
        from src.models.ensemble import AMLEnsembleSystem
        from src.features.feature_extractor import FeatureExtractor
        from src.api.main import app
        
        # Test model initialization
        aml_system = AMLEnsembleSystem()
        feature_extractor = FeatureExtractor()
        
        print("✅ All imports successful")
        print("✅ Models initialize correctly")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def print_next_steps():
    """Print next steps"""
    print("\n🚀 AML 360º Setup Complete!")
    print("\n📋 Next Steps:")
    print("1. Configure your .env file with actual credentials")
    print("2. Set up BigQuery dataset and run SQL scripts:")
    print("   cd sql/ && bq query --use_legacy_sql=false < feature_engineering.sql")
    print("3. Train your models:")
    print("   jupyter lab notebooks/01_AML_EDA_and_Training.ipynb")
    print("4. Start the API server:")
    print("   python -m uvicorn src.api.main:app --reload")
    print("5. Or use Docker:")
    print("   docker-compose -f docker/docker-compose.yml up")
    
    print("\n🔗 Access Points:")
    print("   API: http://localhost:8000")
    print("   API Docs: http://localhost:8000/docs")
    print("   Jupyter: http://localhost:8888")
    print("   Grafana: http://localhost:3000")
    print("   Prometheus: http://localhost:9090")
    
    print("\n📚 Key Files:")
    print("   📖 README.md - Project overview")
    print("   🏗️  docs/technical-architecture.md - Technical details")
    print("   🛠️  config/config.yaml - Configuration")
    print("   🐳 docker/docker-compose.yml - Docker setup")
    print("   📊 notebooks/01_AML_EDA_and_Training.ipynb - Training")

def main():
    """Main setup function"""
    print("🔧 AML 360º System Setup")
    print("=" * 50)
    
    if not check_python_version():
        return
    
    setup_directories()
    create_env_file()
    
    # Optional: Install requirements
    install_req = input("\n📦 Install requirements? (y/N): ").lower() == 'y'
    if install_req:
        if not install_requirements():
            return
    
    # Optional: Run tests
    run_test = input("\n🧪 Run system test? (y/N): ").lower() == 'y'
    if run_test:
        if not quick_test():
            print("⚠️  System test failed - check your environment")
    
    print_next_steps()

if __name__ == "__main__":
    main()
