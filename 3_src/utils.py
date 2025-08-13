"""
Utility functions for the churn prediction project
"""

import os
import pickle
import json
import logging
from datetime import datetime
import pandas as pd
from datasets import load_dataset
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, '1_data')
RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw')
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, 'processed')

# Create directories if they don't exist
os.makedirs(RAW_DATA_PATH, exist_ok=True)
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

def load_from_huggingface(dataset_name="mstz/churn", split="train", force_download=False):
    """
    Load dataset from Hugging Face and save to local CSV
    
    Parameters:
    -----------
    dataset_name : str
        Name of the Hugging Face dataset
    split : str
        Dataset split to load (train/test/validation)
    force_download : bool
        Force re-download even if local file exists
    
    Returns:
    --------
    pd.DataFrame : Loaded dataset as pandas DataFrame
    """
    logger.info(f"Loading dataset: {dataset_name}")
    
    # Check if raw data already exists
    raw_file_path = os.path.join(RAW_DATA_PATH, 'churn_data_raw.csv')
    
    if os.path.exists(raw_file_path) and not force_download:
        logger.info("Loading from existing raw data file")
        return pd.read_csv(raw_file_path)
    
    try:
        # Load from Hugging Face
        logger.info("Downloading from Hugging Face...")
        dataset = load_dataset(dataset_name, split=split)
        
        # Convert to pandas DataFrame
        df = dataset.to_pandas()
        
        # Save to raw data folder
        # Save to raw data folder
        df.to_csv(raw_file_path, index=False)
        logger.info(f"Dataset saved to: {raw_file_path}")
        
        # Save metadata
        metadata = {
            'dataset_name': dataset_name,
            'download_date': datetime.now().isoformat(),
            'split': split,
            'shape': df.shape,
            'columns': list(df.columns)
        }
        
        with open(os.path.join(RAW_DATA_PATH, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        logger.info("Note: If authentication is required, set HF_TOKEN environment variable")
        raise

def save_model(model, filename, path=PROCESSED_DATA_PATH):
    """Save model to pickle file"""
    filepath = os.path.join(path, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to: {filepath}")

def load_model(filename, path=PROCESSED_DATA_PATH):
    """Load model from pickle file"""
    filepath = os.path.join(path, filename)
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from: {filepath}")
    return model

def save_dataframe(df, filename, path=PROCESSED_DATA_PATH):
    """Save DataFrame to CSV"""
    filepath = os.path.join(path, filename)
    df.to_csv(filepath, index=False)
    logger.info(f"DataFrame saved to: {filepath}")

def load_config(config_file='config.json'):
    """Load configuration from JSON file"""
    config_path = os.path.join(PROJECT_ROOT, config_file)
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Utility functions for churn prediction')
    parser.add_argument('--download-data', action='store_true', 
                       help='Download dataset from Hugging Face')
    parser.add_argument('--force-download', action='store_true',
                       help='Force re-download even if data exists')
    
    args = parser.parse_args()
    
    if args.download_data:
        df = load_from_huggingface(force_download=args.force_download)
        print(f"Dataset downloaded: {df.shape}")
        print(f"Columns: {list(df.columns)}")