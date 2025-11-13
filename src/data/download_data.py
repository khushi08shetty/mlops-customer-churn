"""
Script to download and prepare the Telco Customer Churn dataset
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def create_sample_data():
    """
    Create a sample churn dataset for demonstration
    In production, you'd download from a real source
    """
    np.random.seed(42)
    n_samples = 5000
    
    data = {
        'CustomerID': [f'CUST{i:05d}' for i in range(n_samples)],
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples),
        'Tenure': np.random.randint(1, 73, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.35, 0.45, 0.2]),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.25, 0.2]),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        'PaymentMethod': np.random.choice([
            'Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'
        ], n_samples),
        'MonthlyCharges': np.random.uniform(18.0, 118.0, n_samples),
        'TotalCharges': np.random.uniform(18.0, 8500.0, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable with some logic
    churn_prob = (
        (df['Tenure'] < 12) * 0.3 +
        (df['Contract'] == 'Month-to-month') * 0.25 +
        (df['MonthlyCharges'] > 70) * 0.15 +
        np.random.random(n_samples) * 0.3
    )
    df['Churn'] = (churn_prob > 0.5).astype(int)
    
    return df

def main():
    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Generate data
    print("Generating sample data...")
    df = create_sample_data()
    
    # Save raw data
    raw_path = 'data/raw/customer_churn.csv'
    df.to_csv(raw_path, index=False)
    print(f"Raw data saved to {raw_path}")
    print(f"Shape: {df.shape}")
    print(f"\nChurn distribution:")
    print(df['Churn'].value_counts(normalize=True))
    
    # Split data
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['Churn'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['Churn'])
    
    # Save splits
    train_df.to_csv('data/processed/train.csv', index=False)
    val_df.to_csv('data/processed/val.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    
    print(f"\nData split complete:")
    print(f"Train: {train_df.shape[0]} samples")
    print(f"Validation: {val_df.shape[0]} samples")
    print(f"Test: {test_df.shape[0]} samples")

if __name__ == "__main__":
    main()