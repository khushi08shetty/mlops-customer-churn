A complete MLOps implementation for customer churn prediction using open-source tools.

Table of Contents:
Overview
Architecture
Tech Stack
Project Structure
Setup Instructions
Usage
MLOps Components
API Documentation
Testing
Monitoring
CI/CD
Future Improvements

🎯 Overview
This project demonstrates a complete end-to-end MLOps pipeline for predicting customer churn in a telecom company. It includes:
Data versioning with DVC
Experiment tracking with MLflow
Model serving with FastAPI
Containerization with Docker
CI/CD with GitHub Actions
Monitoring with Prometheus & Grafana

🏗️ Architecture
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Data      │────▶│  Training    │────▶│   Model     │
│  Pipeline   │     │   Pipeline   │     │  Registry   │
└─────────────┘     └──────────────┘     └─────────────┘
      │                     │                     │
      │                     ▼                     ▼
      │              ┌──────────────┐     ┌─────────────┐
      └─────────────▶│   MLflow     │     │   FastAPI   │
                     │   Tracking   │     │     API     │
                     └──────────────┘     └─────────────┘
                            │                     │
                            ▼                     ▼
                     ┌──────────────┐     ┌─────────────┐
                     │  Monitoring  │     │   Docker    │
                     │  (Prometheus)│     │  Container  │
                     └──────────────┘     └─────────────┘

🛠️ Tech Stack
Component Technology Purpose
Programming Python 3.9 Core language
MLFramework Scikit-learn Model training
Experiment Tracking MLflowTrack experiments & models
API Framework FastAPI Model serving
Containerization Docker Deployment
Orchestration Apache Airflow Pipeline automation
CI/CD GitHub Actions Automation
Monitoring Prometheus + Grafana Performance tracking
Data Versioning DVC Data & model versioning
Testing Pytest Unit & integration tests

📁 Project Structure
mlops-churn-prediction/
├── data/
│   ├── raw/                    # Raw data
│   └── processed/              # Processed data splits
├── notebooks/
│   └── 01_eda.py              # Exploratory data analysis
├── src/
│   ├── data/
│   │   └── download_data.py   # Data ingestion
│   ├── features/
│   │   └── preprocessing.py   # Feature engineering
│   ├── models/
│   │   └── train.py          # Model training
│   └── api/
│       └── app.py            # FastAPI application
├── tests/
│   ├── test_api.py           # API tests
│   └── test_preprocessing.py # Preprocessing tests
├── monitoring/
│   └── prometheus.yml        # Prometheus config
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml   # CI/CD pipeline
├── docker-compose.yml        # Multi-container setup
├── Dockerfile               # API container
├── requirements.txt         # Python dependencies
└── README.md               # This file
🚀 Setup Instructions
Prerequisites

Python 3.9+
Docker & Docker Compose
Git
(Optional) GitHub account for CI/CD

Step 1: Clone Repository
bashgit clone <your-repo-url>
cd mlops-churn-prediction
Step 2: Create Virtual Environment
bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Step 3: Install Dependencies
bashpip install -r requirements.txt
Step 4: Generate Data
bashpython src/data/download_data.py
Step 5: Preprocess Data
bashpython src/features/preprocessing.py
Step 6: Train Models
bashpython src/models/train.py
Step 7: View MLflow UI
bashmlflow ui --port 5000
Open browser: http://localhost:5000
Step 8: Run API Locally
bashuvicorn src.api.app:app --reload --port 8000
API docs: http://localhost:8000/docs
Step 9: Run with Docker
bashdocker-compose up -d
Services will be available at:

API: http://localhost:8000
MLflow: http://localhost:5000
Prometheus: http://localhost:9090
Grafana: http://localhost:3000 (admin/admin)

💻 Usage
Training Models
bash# Train single model
python src/models/train.py

# Train multiple models for comparison
python src/models/train.py --compare
Making Predictions
Via Python
pythonimport requests

url = "http://localhost:8000/predict"
customer_data = {
    "Gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "Tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.5,
    "TotalCharges": 850.5
}

response = requests.post(url, json=customer_data)
print(response.json())
Via cURL
bashcurl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "Tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.5,
    "TotalCharges": 850.5
  }'
🔬 MLOps Components
1. Data Versioning (DVC)
bash# Initialize DVC
dvc init

# Track data
dvc add data/raw/customer_churn.csv

# Push to remote
dvc push

# Pull data
dvc pull
2. Experiment Tracking (MLflow)

Automatically logs all experiments during training
Tracks parameters, metrics, and artifacts
Compare model performance
Register best models

3. Model Registry

Models stored in MLflow
Version control for models
Stage management (Staging, Production)

4. CI/CD Pipeline

Automated testing on push
Model training on main branch
Docker image building
Deployment automation

5. Monitoring

Prometheus collects metrics
Grafana visualizes performance
Track API latency, throughput
Monitor model predictions

📚 API Documentation
Endpoints
Health Check
GET /health
Single Prediction
POST /predict
Body: Customer object
Response: {
  "churn_prediction": 0 or 1,
  "churn_probability": 0.XX,
  "risk_level": "Low/Medium/High"
}
Batch Prediction
POST /batch_predict
Body: {customers: [Customer objects]}
Response: {predictions: [predictions]}
Access interactive docs at: http://localhost:8000/docs
🧪 Testing
Run All Tests
bashpytest tests/ -v
Run with Coverage
bashpytest tests/ --cov=src --cov-report=html
Run Specific Test File
bashpytest tests/test_api.py -v
📊 Monitoring
Prometheus Metrics

API request count
Response time
Error rates
Model prediction distribution

Grafana Dashboards

API Performance Dashboard
Model Performance Dashboard
System Health Dashboard

🔄 CI/CD Pipeline
The GitHub Actions workflow:

Test Stage: Runs linting and unit tests
Build Stage: Creates Docker image
Train Stage: Trains model on new data
Deploy Stage: Deploys to production

📈 Model Performance
Current best model metrics (Random Forest):

Accuracy: 0.XX
Precision: 0.XX
Recall: 0.XX
F1 Score: 0.XX
ROC-AUC: 0.XX

🔮 Future Improvements

 Add A/B testing framework
 Implement model drift detection
 Add feature store
 Integrate with Kubernetes
 Add data quality checks
 Implement model explainability
 Add real-time streaming predictions
 Set up alerting system

📝 Documentation for Submission
Project Report Structure

Introduction

Problem statement
Objectives
Scope


System Architecture

Architecture diagram
Component description
Data flow


Implementation

Tools and technologies
Setup instructions
Code walkthrough


MLOps Pipeline

Data versioning
Experiment tracking
Model registry
CI/CD process
Monitoring setup


Results

Model performance
API performance
Screenshots of dashboards


Challenges & Solutions

Problems faced
How they were solved


Conclusion & Future Work

Screenshots to Include

MLflow experiment tracking
Model comparison dashboard
API documentation (Swagger UI)
Prometheus metrics
Grafana dashboards
GitHub Actions pipeline
Docker containers running

🤝 Contributing

Fork the repository
Create feature branch
Commit changes
Push to branch
Open pull request

📄 License
MIT License
👥 Author
Your Name - [Your Email]
🙏 Acknowledgments

Dataset: Telco Customer Churn
Inspired by MLOps best practices
Open-source community

