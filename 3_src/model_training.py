"""
Model training and evaluation functions
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import logging
import joblib
from .utils import save_model, load_model, save_dataframe, PROCESSED_DATA_PATH

logger = logging.getLogger(__name__)

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def train_logistic_regression(X_train, y_train, use_smote=True):
    """Train Logistic Regression model"""
    logger.info("Training Logistic Regression...")
    
    if use_smote:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
    
    # Hyperparameter tuning
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['liblinear']
    }
    
    lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train_balanced, y_train_balanced)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def train_random_forest(X_train, y_train, use_smote=True):
    """Train Random Forest model"""
    logger.info("Training Random Forest...")
    
    if use_smote:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train_balanced, y_train_balanced)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def train_xgboost(X_train, y_train, use_smote=True):
    """Train XGBoost model"""
    logger.info("Training XGBoost...")
    
    if use_smote:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0]
    }
    
    xgb = XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss')
    grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train_balanced, y_train_balanced)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    logger.info(f"Evaluating {model_name}...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    logger.info(f"{model_name} Results:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    
    return metrics

def get_feature_importance(model, feature_names, model_name):
    """Extract feature importance from model"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        return None
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
        'model': model_name
    }).sort_values('importance', ascending=False)
    
    return feature_importance

def train_models(X_train, y_train, X_test, y_test, feature_names):
    """Train all models and compare performance"""
    logger.info("Starting model training pipeline...")
    
    models = {
        'Logistic Regression': train_logistic_regression(X_train, y_train),
        'Random Forest': train_random_forest(X_train, y_train),
        'XGBoost': train_xgboost(X_train, y_train)
    }
    
    results = []
    feature_importances = []
    
    for model_name, model in models.items():
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test, model_name)
        results.append(metrics)
        
        # Get feature importance
        feat_imp = get_feature_importance(model, feature_names, model_name)
        if feat_imp is not None:
            feature_importances.append(feat_imp)
        
        # Save model
        save_model(model, f'{model_name.lower().replace(" ", "_")}_model.pkl')
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    save_dataframe(results_df, 'model_comparison_results.csv')
    
    # Save feature importances
    if feature_importances:
        all_importances = pd.concat(feature_importances)
        save_dataframe(all_importances, 'feature_importances.csv')
    
    # Select best model based on ROC-AUC
    best_model_idx = results_df['roc_auc'].idxmax()
    best_model_name = results_df.loc[best_model_idx, 'model_name']
    best_model = models[best_model_name]
    
    logger.info(f"\nBest model: {best_model_name} with ROC-AUC: {results_df.loc[best_model_idx, 'roc_auc']:.4f}")
    
    # Save best model
    save_model(best_model, 'best_model.pkl')
    
    # Generate predictions for dashboard
    predictions = best_model.predict_proba(X_test)[:, 1]
    pred_df = pd.DataFrame({
        'customer_index': X_test.index,
        'churn_probability': predictions,
        'predicted_churn': (predictions > 0.5).astype(int),
        'actual_churn': y_test.values
    })
    save_dataframe(pred_df, 'test_predictions.csv')
    
    return models, results_df, best_model

if __name__ == "__main__":
    # Load processed data
    import os
    X_train = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'y_train.csv'))['Churn']
    y_test = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'y_test.csv'))['Churn']
    
    feature_names = list(X_train.columns)
    
    # Train models
    models, results, best_model = train_models(X_train, y_train, X_test, y_test, feature_names)
    
    print("\nModel training completed!")
    print(results)