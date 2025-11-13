"""
Model training with MLflow experiment tracking
"""
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.preprocessing import prepare_data


def evaluate_model(model, X, y, dataset_name=""):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    metrics = {
        f'{dataset_name}_accuracy': accuracy_score(y, y_pred),
        f'{dataset_name}_precision': precision_score(y, y_pred),
        f'{dataset_name}_recall': recall_score(y, y_pred),
        f'{dataset_name}_f1': f1_score(y, y_pred),
        f'{dataset_name}_roc_auc': roc_auc_score(y, y_pred_proba),
    }
    
    return metrics, y_pred, y_pred_proba


def train_model(model_type='random_forest', experiment_name='churn_prediction'):
    """
    Train a model with MLflow tracking
    """
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    # Load and prepare data
    print("Preparing data...")
    (X_train, y_train, 
     X_val, y_val, 
     X_test, y_test, 
     preprocessor) = prepare_data(
        'data/processed/train.csv',
        'data/processed/val.csv',
        'data/processed/test.csv'
    )
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"{model_type}_run"):
        
        # Define model
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5
            }
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            params = {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1
            }
        else:  # logistic_regression
            model = LogisticRegression(
                max_iter=1000,
                random_state=42
            )
            params = {
                'max_iter': 1000,
                'solver': 'lbfgs'
            }
        
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param('model_type', model_type)
        
        # Train model
        print(f"Training {model_type}...")
        model.fit(X_train, y_train)
        
        # Evaluate on train set
        train_metrics, _, _ = evaluate_model(model, X_train, y_train, 'train')
        
        # Evaluate on validation set
        val_metrics, val_pred, val_pred_proba = evaluate_model(model, X_val, y_val, 'val')
        
        # Evaluate on test set
        test_metrics, test_pred, test_pred_proba = evaluate_model(model, X_test, y_test, 'test')
        
        # Combine all metrics
        all_metrics = {**train_metrics, **val_metrics, **test_metrics}
        
        # Log metrics
        mlflow.log_metrics(all_metrics)
        
        # Log confusion matrix
        cm = confusion_matrix(y_test, test_pred)
        print("\nConfusion Matrix (Test Set):")
        print(cm)
        
        # Log classification report
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, test_pred))
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally
        model_path = f'models/{model_type}_model.pkl'
        import joblib
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        
        print(f"\nModel Metrics:")
        print(f"Test Accuracy: {test_metrics['test_accuracy']:.4f}")
        print(f"Test F1 Score: {test_metrics['test_f1']:.4f}")
        print(f"Test ROC-AUC: {test_metrics['test_roc_auc']:.4f}")
        
        return model, all_metrics


def train_multiple_models():
    """Train multiple models and compare"""
    models = ['logistic_regression', 'random_forest', 'gradient_boosting']
    results = {}
    
    for model_type in models:
        print(f"\n{'='*60}")
        print(f"Training {model_type}")
        print(f"{'='*60}")
        model, metrics = train_model(model_type)
        results[model_type] = metrics
    
    # Compare results
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df[['test_accuracy', 'test_precision', 
                                    'test_recall', 'test_f1', 'test_roc_auc']]
    print(comparison_df.round(4))
    
    # Find best model
    best_model = comparison_df['test_f1'].idxmax()
    print(f"\nBest Model: {best_model}")
    print(f"Best F1 Score: {comparison_df.loc[best_model, 'test_f1']:.4f}")
    
    return results


if __name__ == "__main__":
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Train multiple models
    results = train_multiple_models()