"""
Customer Churn Prediction Package
A comprehensive solution for predicting and preventing customer churn
"""

__version__ = "1.0.0"
__author__ = "Data Science Team"

from .data_preprocessing import load_from_huggingface, preprocess_data
from .model_training import train_models, evaluate_model
from .visualization import create_churn_plots
from .utils import save_model, load_model

__all__ = [
    'load_from_huggingface',
    'preprocess_data',
    'train_models',
    'evaluate_model',
    'create_churn_plots',
    'save_model',
    'load_model'
]