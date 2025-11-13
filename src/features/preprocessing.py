"""
Feature engineering and preprocessing pipeline
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os

class ChurnPreprocessor(BaseEstimator, TransformerMixin):
    """Custom preprocessor for churn prediction"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = None
        self.numeric_features = ['Tenure', 'MonthlyCharges', 'TotalCharges']
        self.categorical_features = [
            'Gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod',
            'TenureGroup',  # Add engineered categorical here
            'ChargeGroup'  # Add engineered categorical here
        ]
        
    def fit(self, X, y=None):
        """Fit the preprocessor"""
        X = X.copy()
    
        # Fit scalers for numeric features
        for col in self.numeric_features:
            if col in X.columns:
                scaler = StandardScaler()
                self.scalers[col] = scaler.fit(X[[col]])
        
        # Fit encoders for categorical features
        for col in self.categorical_features:
            if col in X.columns:
                encoder = LabelEncoder()
                self.encoders[col] = encoder.fit(X[col].astype(str))
        
        return self
    
    def transform(self, X):
        """Transform the data"""
        X = X.copy()
        
        # Drop CustomerID if present
        if 'CustomerID' in X.columns:
            X = X.drop('CustomerID', axis=1)
        
        # Scale numeric features
        for col in self.numeric_features:
            if col in X.columns and col in self.scalers:
                X[col] = self.scalers[col].transform(X[[col]])
        
        # Encode categorical features
        for col in self.categorical_features:
            if col in X.columns and col in self.encoders:
                X[col] = self.encoders[col].transform(X[col].astype(str))
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        return X
    
    def save(self, path):
        """Save the preprocessor"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Preprocessor saved to {path}")
    
    @staticmethod
    def load(path):
        """Load the preprocessor"""
        return joblib.load(path)


def create_features(df):
    """Create additional features"""
    df = df.copy()
    
    # Feature engineering
    df['AvgChargePerMonth'] = df['TotalCharges'] / (df['Tenure'] + 1)
    df['TenureGroup'] = pd.cut(df['Tenure'], bins=[0, 12, 24, 48, 72], 
                                labels=['0-12', '12-24', '24-48', '48-72'])
    df['ChargeGroup'] = pd.cut(df['MonthlyCharges'], bins=[0, 35, 70, 120], 
                                labels=['Low', 'Medium', 'High'])
    
    # Convert categorical features to string
    df['TenureGroup'] = df['TenureGroup'].astype(str)
    df['ChargeGroup'] = df['ChargeGroup'].astype(str)
    
    return df


def prepare_data(train_path, val_path, test_path, save_preprocessor=True):
    """
    Load and prepare data for modeling
    """
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    print("Creating features...")
    train_df = create_features(train_df)
    val_df = create_features(val_df)
    test_df = create_features(test_df)
    
    # Separate features and target
    X_train = train_df.drop('Churn', axis=1)
    y_train = train_df['Churn']
    
    X_val = val_df.drop('Churn', axis=1)
    y_val = val_df['Churn']
    
    X_test = test_df.drop('Churn', axis=1)
    y_test = test_df['Churn']
    
    # Initialize and fit preprocessor
    print("Fitting preprocessor...")
    preprocessor = ChurnPreprocessor()
    preprocessor.fit(X_train)
    
    # Transform data
    print("Transforming data...")
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    if save_preprocessor:
        preprocessor.save('models/preprocessor.pkl')
    
    print(f"Processed data shapes:")
    print(f"Train: {X_train_processed.shape}")
    print(f"Val: {X_val_processed.shape}")
    print(f"Test: {X_test_processed.shape}")
    
    return (X_train_processed, y_train, 
            X_val_processed, y_val, 
            X_test_processed, y_test, 
            preprocessor)


if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    prepare_data('data/processed/train.csv', 
                 'data/processed/val.csv', 
                 'data/processed/test.csv')