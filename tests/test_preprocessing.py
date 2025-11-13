"""
Tests for preprocessing pipeline
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.preprocessing import ChurnPreprocessor, create_features

def test_create_features():
    """Test feature creation"""
    df = pd.DataFrame({
        'Tenure': [12, 24, 36],
        'MonthlyCharges': [50.0, 75.0, 100.0],
        'TotalCharges': [600.0, 1800.0, 3600.0]
    })
    
    result = create_features(df)
    
    assert 'AvgChargePerMonth' in result.columns
    assert 'TenureGroup' in result.columns
    assert 'ChargeGroup' in result.columns
    assert len(result) == len(df)

def test_preprocessor_fit_transform():
    """Test preprocessor fit and transform"""
    df = pd.DataFrame({
        'CustomerID': ['C001', 'C002', 'C003'],
        'Gender': ['Male', 'Female', 'Male'],
        'SeniorCitizen': [0, 1, 0],
        'Partner': ['Yes', 'No', 'Yes'],
        'Dependents': ['No', 'Yes', 'No'],
        'Tenure': [12, 24, 36],
        'PhoneService': ['Yes', 'Yes', 'No'],
        'MultipleLines': ['No', 'Yes', 'No'],
        'InternetService': ['DSL', 'Fiber optic', 'DSL'],
        'OnlineSecurity': ['Yes', 'No', 'Yes'],
        'OnlineBackup': ['No', 'Yes', 'No'],
        'DeviceProtection': ['Yes', 'No', 'Yes'],
        'TechSupport': ['No', 'Yes', 'No'],
        'StreamingTV': ['Yes', 'No', 'Yes'],
        'StreamingMovies': ['No', 'Yes', 'No'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaperlessBilling': ['Yes', 'No', 'Yes'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer'],
        'MonthlyCharges': [50.0, 75.0, 100.0],
        'TotalCharges': [600.0, 1800.0, 3600.0]
    })
    
    preprocessor = ChurnPreprocessor()
    preprocessor.fit(df)
    
    transformed = preprocessor.transform(df)
    
    assert isinstance(transformed, pd.DataFrame)
    assert 'CustomerID' not in transformed.columns
    assert len(transformed) == len(df)
    assert all(transformed.dtypes.apply(lambda x: np.issubdtype(x, np.number)))

def test_preprocessor_consistency():
    """Test that preprocessor produces consistent results"""
    df = pd.DataFrame({
        'CustomerID': ['C001'],
        'Gender': ['Male'],
        'SeniorCitizen': [0],
        'Partner': ['Yes'],
        'Dependents': ['No'],
        'Tenure': [12],
        'PhoneService': ['Yes'],
        'MultipleLines': ['No'],
        'InternetService': ['DSL'],
        'OnlineSecurity': ['Yes'],
        'OnlineBackup': ['No'],
        'DeviceProtection': ['Yes'],
        'TechSupport': ['No'],
        'StreamingTV': ['Yes'],
        'StreamingMovies': ['No'],
        'Contract': ['Month-to-month'],
        'PaperlessBilling': ['Yes'],
        'PaymentMethod': ['Electronic check'],
        'MonthlyCharges': [50.0],
        'TotalCharges': [600.0]
    })
    
    preprocessor = ChurnPreprocessor()
    preprocessor.fit(df)
    
    result1 = preprocessor.transform(df)
    result2 = preprocessor.transform(df)
    
    pd.testing.assert_frame_equal(result1, result2)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])