"""
Exploratory Data Analysis for Customer Churn
Save this as a .py file or convert to .ipynb
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data/raw/customer_churn.csv')

print("=" * 50)
print("DATA OVERVIEW")
print("=" * 50)
print(f"\nDataset Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

print("\n" + "=" * 50)
print("DATA TYPES & MISSING VALUES")
print("=" * 50)
print(df.info())
print(f"\nMissing values:\n{df.isnull().sum()}")

print("\n" + "=" * 50)
print("NUMERICAL FEATURES STATISTICS")
print("=" * 50)
print(df.describe())

print("\n" + "=" * 50)
print("TARGET VARIABLE DISTRIBUTION")
print("=" * 50)
print(df['Churn'].value_counts())
print(f"\nChurn Rate: {df['Churn'].mean():.2%}")

print("\n" + "=" * 50)
print("CATEGORICAL FEATURES")
print("=" * 50)
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col != 'CustomerID':
        print(f"\n{col}:")
        print(df[col].value_counts())

# Visualizations would go here in a notebook
# For now, save key insights
insights = {
    'total_customers': len(df),
    'churn_rate': df['Churn'].mean(),
    'avg_tenure': df['Tenure'].mean(),
    'avg_monthly_charges': df['MonthlyCharges'].mean(),
}

print("\n" + "=" * 50)
print("KEY INSIGHTS")
print("=" * 50)
for key, value in insights.items():
    print(f"{key}: {value:.2f}")