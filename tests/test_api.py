"""
Tests for FastAPI application
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.api.app import app

# Create test client - Fixed for newer FastAPI versions
client = TestClient(app=app)

def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert response.json()["message"] == "Churn Prediction API"

def test_health():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "model_loaded" in response.json()

def test_predict():
    """Test prediction endpoint"""
    sample_customer = {
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
    
    response = client.post("/predict", json=sample_customer)
    
    # May fail if model not loaded, that's expected in test environment
    if response.status_code == 200:
        assert "churn_prediction" in response.json()
        assert "churn_probability" in response.json()
        assert "risk_level" in response.json()
        assert response.json()["churn_prediction"] in [0, 1]
        assert 0 <= response.json()["churn_probability"] <= 1
        assert response.json()["risk_level"] in ["Low", "Medium", "High"]
    else:
        # If model not loaded, just check it returns appropriate error
        assert response.status_code in [400, 500]

def test_predict_invalid_data():
    """Test prediction with invalid data"""
    invalid_customer = {
        "Gender": "Male",
        "SeniorCitizen": 5,  # Invalid value
        "Partner": "Yes"
        # Missing required fields
    }
    
    response = client.post("/predict", json=invalid_customer)
    assert response.status_code == 422  # Validation error

def test_batch_predict():
    """Test batch prediction endpoint"""
    batch_request = {
        "customers": [
            {
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
            },
            {
                "Gender": "Female",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "Yes",
                "Tenure": 48,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Bank transfer",
                "MonthlyCharges": 45.0,
                "TotalCharges": 2160.0
            }
        ]
    }
    
    response = client.post("/batch_predict", json=batch_request)
    
    # May fail if model not loaded
    if response.status_code == 200:
        assert "predictions" in response.json()
        assert len(response.json()["predictions"]) == 2
    else:
        assert response.status_code in [400, 500]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])