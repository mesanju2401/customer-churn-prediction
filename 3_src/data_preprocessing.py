"""
Data preprocessing functions for churn prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import logging
from .utils import load_from_huggingface, save_dataframe, RAW_DATA_PATH, PROCESSED_DATA_PATH

logger = logging.getLogger(__name__)

def clean_data(df):
    """
    Clean raw data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataframe
        
    Returns:
    --------
    pd.DataFrame : Cleaned dataframe
    """
    logger.info("Starting data cleaning...")
    df = df.copy()
    
    # Handle TotalCharges - convert empty strings to NaN and then to numeric
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    
    # Fill missing TotalCharges with monthly charges for new customers
    mask = df['TotalCharges'].isna()
    df.loc[mask, 'TotalCharges'] = df.loc[mask, 'MonthlyCharges']
    
    # Convert SeniorCitizen to Yes/No
    # Convert SeniorCitizen to Yes/No
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    
    # Remove customerID for modeling (keep for reference)
    df['customerID'] = df['customerID'].astype(str)
    
    logger.info(f"Data cleaned. Shape: {df.shape}")
    logger.info(f"Missing values: {df.isnull().sum().sum()}")
    
    return df

def engineer_features(df):
    """
    Create new features for better model performance
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataframe
        
    Returns:
    --------
    pd.DataFrame : Dataframe with engineered features
    """
    logger.info("Engineering features...")
    df = df.copy()
    
    # Tenure groups
    df['tenure_group'] = pd.cut(df['tenure'], 
                                bins=[0, 12, 24, 48, 72], 
                                labels=['0-12', '12-24', '24-48', '48-72'])
    
    # Average charges per month (handling zero tenure)
    df['avg_monthly_charge'] = np.where(df['tenure'] > 0, 
                                        df['TotalCharges'] / df['tenure'], 
                                        df['MonthlyCharges'])
    
    # Total services count
    services = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['services_count'] = (df[services] == 'Yes').sum(axis=1)
    
    # Has streaming services
    df['has_streaming'] = ((df['StreamingTV'] == 'Yes') | 
                          (df['StreamingMovies'] == 'Yes')).astype(int)
    
    # Payment type risk (Electronic check has higher churn)
    df['high_risk_payment'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    
    # Contract type risk
    df['month_to_month'] = (df['Contract'] == 'Month-to-month').astype(int)
    
    # No online services
    online_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    df['no_online_services'] = (df[online_services] == 'No').sum(axis=1)
    
    logger.info(f"Features engineered. New shape: {df.shape}")
    
    return df

def encode_categorical(df, target_column='Churn'):
    """
    Encode categorical variables
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with categorical variables
    target_column : str
        Name of target column
        
    Returns:
    --------
    pd.DataFrame : Encoded dataframe
    dict : Mapping of encoders used
    """
    logger.info("Encoding categorical variables...")
    df = df.copy()
    encoders = {}
    
    # Binary columns - simple mapping
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                   'PaperlessBilling', 'Churn', 'SeniorCitizen']
    
    for col in binary_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col])
            
            #there is an error i guess here 
            encoders[col] = le
    
    # Multi-class categorical columns - one-hot encoding
    categorical_cols = ['InternetService', 'Contract', 'PaymentMethod', 'tenure_group']
    
    for col in categorical_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
    
    # Handle service-related columns with 'No internet service' values
    service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    for col in service_cols:
        if col in df.columns:
            # Convert 'No internet service' to 'No'
            df[col] = df[col].replace('No internet service', 'No')
            df[col] = df[col].replace('No phone service', 'No')
            df[col + '_encoded'] = LabelEncoder().fit_transform(df[col])
    
    logger.info(f"Encoding complete. Shape: {df.shape}")
    
    return df, encoders

def preprocess_data(df=None, save_processed=True):
    """
    Complete preprocessing pipeline
    
    Parameters:
    -----------
    df : pd.DataFrame, optional
        Input dataframe. If None, loads from Hugging Face
    save_processed : bool
        Whether to save processed data
        
    Returns:
    --------
    dict : Dictionary containing processed data splits and encoders
    """
    if df is None:
        df = load_from_huggingface()
    
    # Clean data
    df_clean = clean_data(df)
    
    # Engineer features
    df_features = engineer_features(df_clean)
    
    # Encode categorical variables
    df_encoded, encoders = encode_categorical(df_features)
    
    # Select features for modeling
    feature_cols = [col for col in df_encoded.columns 
                   if col.endswith('_encoded') or col in ['tenure', 'MonthlyCharges', 
                   'TotalCharges', 'avg_monthly_charge', 'services_count', 
                   'has_streaming', 'high_risk_payment', 'month_to_month', 
                   'no_online_services'] or col.startswith(('InternetService_', 
                   'Contract_', 'PaymentMethod_', 'tenure_group_'))]
    
    # Prepare modeling data
    X = df_encoded[feature_cols]
    y = df_encoded['Churn_encoded']
    
    # Keep customer info for later analysis
    customer_info = df_encoded[['customerID', 'Churn']]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
    
    if save_processed:
        # Save processed data
        save_dataframe(df_clean, 'churn_data_cleaned.csv')
        save_dataframe(df_encoded, 'churn_data_processed.csv')
        save_dataframe(X_train_scaled, 'X_train.csv')
        save_dataframe(X_test_scaled, 'X_test.csv')
        save_dataframe(pd.DataFrame(y_train, columns=['Churn']), 'y_train.csv')
        save_dataframe(pd.DataFrame(y_test, columns=['Churn']), 'y_test.csv')
        
        # Save encoders and scaler
        import pickle
        with open(os.path.join(PROCESSED_DATA_PATH, 'encoders.pkl'), 'wb') as f:
            pickle.dump(encoders, f)
        with open(os.path.join(PROCESSED_DATA_PATH, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
    
    logger.info("Preprocessing complete!")
    logger.info(f"Train set: {X_train_scaled.shape}")
    logger.info(f"Test set: {X_test_scaled.shape}")
    logger.info(f"Churn rate - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_cols,
        'encoders': encoders,
        'scaler': scaler,
        'full_data': df_encoded,
        'customer_info': customer_info
    }

if __name__ == "__main__":
    # Run preprocessing pipeline
    processed_data = preprocess_data()
    print("Preprocessing completed successfully!")