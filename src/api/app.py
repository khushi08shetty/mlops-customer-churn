# """
# FastAPI application for churn prediction
# """
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, Field, ConfigDict
# import joblib
# import pandas as pd
# import numpy as np
# from typing import List, Optional
# import os
# import sys

# # Add parent directory to path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from features.preprocessing import ChurnPreprocessor

# # Initialize FastAPI app
# app = FastAPI(
#     title="Churn Prediction API",
#     description="API for predicting customer churn",
#     version="1.0.0"
# )

# # Load model and preprocessor
# MODEL_PATH = "models/random_forest_model.pkl"
# PREPROCESSOR_PATH = "models/preprocessor.pkl"

# try:
#     model = joblib.load(MODEL_PATH)
#     preprocessor = joblib.load(PREPROCESSOR_PATH)
#     print("Model and preprocessor loaded successfully")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     model = None
#     preprocessor = None


# # Pydantic models for request/response
# class Customer(BaseModel):
#     """Customer data schema"""
#     model_config = ConfigDict(
#         json_schema_extra={
#             "example": {
#                 "Gender": "Male",
#                 "SeniorCitizen": 0,
#                 "Partner": "Yes",
#                 "Dependents": "No",
#                 "Tenure": 12,
#                 "PhoneService": "Yes",
#                 "MultipleLines": "No",
#                 "InternetService": "Fiber optic",
#                 "OnlineSecurity": "No",
#                 "OnlineBackup": "Yes",
#                 "DeviceProtection": "No",
#                 "TechSupport": "No",
#                 "StreamingTV": "Yes",
#                 "StreamingMovies": "No",
#                 "Contract": "Month-to-month",
#                 "PaperlessBilling": "Yes",
#                 "PaymentMethod": "Electronic check",
#                 "MonthlyCharges": 70.5,
#                 "TotalCharges": 850.5
#             }
#         }
#     )
    
#     Gender: str
#     SeniorCitizen: int = Field(..., ge=0, le=1)
#     Partner: str
#     Dependents: str
#     Tenure: int = Field(..., ge=0)
#     PhoneService: str
#     MultipleLines: str
#     InternetService: str
#     OnlineSecurity: str
#     OnlineBackup: str
#     DeviceProtection: str
#     TechSupport: str
#     StreamingTV: str
#     StreamingMovies: str
#     Contract: str
#     PaperlessBilling: str
#     PaymentMethod: str
#     MonthlyCharges: float = Field(..., gt=0)
#     TotalCharges: float = Field(..., gt=0)


# class PredictionResponse(BaseModel):
#     """Prediction response schema"""
#     churn_prediction: int
#     churn_probability: float
#     risk_level: str


# class BatchPredictionRequest(BaseModel):
#     """Batch prediction request schema"""
#     customers: List[Customer]


# class HealthResponse(BaseModel):
#     """Health check response"""
#     model_config = ConfigDict(protected_namespaces=())
    
#     status: str
#     model_loaded: bool


# @app.get("/", response_model=dict)
# async def root():
#     """Root endpoint"""
#     return {
#         "message": "Churn Prediction API",
#         "version": "1.0.0",
#         "endpoints": {
#             "health": "/health",
#             "predict": "/predict",
#             "batch_predict": "/batch_predict",
#             "docs": "/docs"
#         }
#     }


# @app.get("/health", response_model=HealthResponse)
# async def health():
#     """Health check endpoint"""
#     return {
#         "status": "healthy" if model is not None else "unhealthy",
#         "model_loaded": model is not None
#     }


# @app.post("/predict", response_model=PredictionResponse)
# async def predict(customer: Customer):
#     """
#     Predict churn for a single customer
#     """
#     if model is None or preprocessor is None:
#         raise HTTPException(status_code=500, detail="Model not loaded")
    
#     try:
#         # Convert to DataFrame
#         customer_dict = customer.model_dump()
#         customer_dict['CustomerID'] = 'TEMP001'  # Temporary ID
#         df = pd.DataFrame([customer_dict])
        
#         # Add engineered features
#         df['AvgChargePerMonth'] = df['TotalCharges'] / (df['Tenure'] + 1)
#         df['TenureGroup'] = pd.cut(df['Tenure'], bins=[0, 12, 24, 48, 72], 
#                                     labels=['0-12', '12-24', '24-48', '48-72'])
#         df['ChargeGroup'] = pd.cut(df['MonthlyCharges'], bins=[0, 35, 70, 120], 
#                                     labels=['Low', 'Medium', 'High'])
#         df['TenureGroup'] = df['TenureGroup'].astype(str)
#         df['ChargeGroup'] = df['ChargeGroup'].astype(str)
        
#         # Preprocess
#         X = preprocessor.transform(df)
        
#         # Predict
#         prediction = int(model.predict(X)[0])
#         probability = float(model.predict_proba(X)[0][1])
        
#         # Determine risk level
#         if probability < 0.3:
#             risk_level = "Low"
#         elif probability < 0.6:
#             risk_level = "Medium"
#         else:
#             risk_level = "High"
        
#         return {
#             "churn_prediction": prediction,
#             "churn_probability": round(probability, 4),
#             "risk_level": risk_level
#         }
    
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


# @app.post("/batch_predict")
# async def batch_predict(request: BatchPredictionRequest):
#     """
#     Predict churn for multiple customers
#     """
#     if model is None or preprocessor is None:
#         raise HTTPException(status_code=500, detail="Model not loaded")
    
#     try:
#         predictions = []
        
#         for customer in request.customers:
#             # Use the single prediction endpoint logic
#             customer_dict = customer.model_dump()
#             customer_dict['CustomerID'] = 'TEMP'
#             df = pd.DataFrame([customer_dict])
            
#             # Add engineered features
#             df['AvgChargePerMonth'] = df['TotalCharges'] / (df['Tenure'] + 1)
#             df['TenureGroup'] = pd.cut(df['Tenure'], bins=[0, 12, 24, 48, 72], 
#                                         labels=['0-12', '12-24', '24-48', '48-72'])
#             df['ChargeGroup'] = pd.cut(df['MonthlyCharges'], bins=[0, 35, 70, 120], 
#                                         labels=['Low', 'Medium', 'High'])
#             df['TenureGroup'] = df['TenureGroup'].astype(str)
#             df['ChargeGroup'] = df['ChargeGroup'].astype(str)
            
#             X = preprocessor.transform(df)
#             prediction = int(model.predict(X)[0])
#             probability = float(model.predict_proba(X)[0][1])
            
#             if probability < 0.3:
#                 risk_level = "Low"
#             elif probability < 0.6:
#                 risk_level = "Medium"
#             else:
#                 risk_level = "High"
            
#             predictions.append({
#                 "churn_prediction": prediction,
#                 "churn_probability": round(probability, 4),
#                 "risk_level": risk_level
#             })
        
#         return {"predictions": predictions}
    
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
"""
FastAPI application for churn prediction
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, ConfigDict
import joblib
import pandas as pd
import numpy as np
from typing import List, Optional
import os
import sys
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.preprocessing import ChurnPreprocessor

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="API for predicting customer churn",
    version="1.0.0"
)

# Load model and preprocessor
MODEL_PATH = "models/random_forest_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print("Model and preprocessor loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    preprocessor = None


# Pydantic models for request/response
class Customer(BaseModel):
    """Customer data schema"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )
    
    Gender: str
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str
    Dependents: str
    Tenure: int = Field(..., ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(..., gt=0)
    TotalCharges: float = Field(..., gt=0)


class PredictionResponse(BaseModel):
    """Prediction response schema"""
    churn_prediction: int
    churn_probability: float
    risk_level: str


class BatchPredictionRequest(BaseModel):
    """Batch prediction request schema"""
    customers: List[Customer]


class HealthResponse(BaseModel):
    """Health check response"""
    model_config = ConfigDict(protected_namespaces=())
    
    status: str
    model_loaded: bool


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: Customer):
    """
    Predict churn for a single customer
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        customer_dict = customer.model_dump()
        customer_dict['CustomerID'] = 'TEMP001'  # Temporary ID
        df = pd.DataFrame([customer_dict])
        
        # Add engineered features
        df['AvgChargePerMonth'] = df['TotalCharges'] / (df['Tenure'] + 1)
        df['TenureGroup'] = pd.cut(df['Tenure'], bins=[0, 12, 24, 48, 72], 
                                    labels=['0-12', '12-24', '24-48', '48-72'])
        df['ChargeGroup'] = pd.cut(df['MonthlyCharges'], bins=[0, 35, 70, 120], 
                                    labels=['Low', 'Medium', 'High'])
        df['TenureGroup'] = df['TenureGroup'].astype(str)
        df['ChargeGroup'] = df['ChargeGroup'].astype(str)
        
        # Preprocess
        X = preprocessor.transform(df)
        
        # Predict
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return {
            "churn_prediction": prediction,
            "churn_probability": round(probability, 4),
            "risk_level": risk_level
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/batch_predict")
async def batch_predict(request: BatchPredictionRequest):
    """
    Predict churn for multiple customers
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        predictions = []
        
        for customer in request.customers:
            # Use the single prediction endpoint logic
            customer_dict = customer.model_dump()
            customer_dict['CustomerID'] = 'TEMP'
            df = pd.DataFrame([customer_dict])
            
            # Add engineered features
            df['AvgChargePerMonth'] = df['TotalCharges'] / (df['Tenure'] + 1)
            df['TenureGroup'] = pd.cut(df['Tenure'], bins=[0, 12, 24, 48, 72], 
                                        labels=['0-12', '12-24', '24-48', '48-72'])
            df['ChargeGroup'] = pd.cut(df['MonthlyCharges'], bins=[0, 35, 70, 120], 
                                        labels=['Low', 'Medium', 'High'])
            df['TenureGroup'] = df['TenureGroup'].astype(str)
            df['ChargeGroup'] = df['ChargeGroup'].astype(str)
            
            X = preprocessor.transform(df)
            prediction = int(model.predict(X)[0])
            probability = float(model.predict_proba(X)[0][1])
            
            if probability < 0.3:
                risk_level = "Low"
            elif probability < 0.6:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            predictions.append({
                "churn_prediction": prediction,
                "churn_probability": round(probability, 4),
                "risk_level": risk_level
            })
        
        return {"predictions": predictions}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)