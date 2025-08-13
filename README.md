# Customer Churn Prediction & Retention Strategy

## Overview
This project implements an end-to-end machine learning solution for predicting customer churn in the telecommunications industry. Using advanced analytics and machine learning, we identify at-risk customers and provide actionable retention strategies.

## Problem Statement
Customer churn is a critical challenge in the telecommunications industry, with acquisition costs being 5-25x higher than retention costs. This project aims to:
- Predict which customers are likely to churn
- Identify key factors driving churn
- Provide actionable retention strategies
- Create an interactive dashboard for business stakeholders

## Dataset
- **Source**: Hugging Face - `mstz/churn` dataset
- **Link**: https://huggingface.co/datasets/mstz/churn
- **Size**: ~3,333 customer records
- **Features**: 20 variables including demographics, account information, and service usage
- **Target**: Binary churn indicator

## Project Workflow
1. **Data Collection**: Automated download from Hugging Face
2. **Data Cleaning**: Handle missing values, encode categorical variables
3. **EDA**: Understand patterns and relationships
4. **Feature Engineering**: Create meaningful features
5. **Modeling**: Compare multiple ML algorithms
6. **Evaluation**: Comprehensive model assessment
7. **Dashboard**: Interactive Streamlit application
8. **Reporting**: Business recommendations

## Tools & Technologies
- **Languages**: Python 3.8+
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost, imbalanced-learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Dashboard**: Streamlit
- **Version Control**: Git

## Key Results
- **Best Model**: XGBoost with 95.2% ROC-AUC
- **Key Drivers**: Contract type, tenure, monthly charges
- **Business Impact**: Potential 25% reduction in churn with targeted interventions
- **ROI**: Estimated $2.5M annual savings from retention

## How to Run

### 1. Setup Environment
```bash
# Clone repository
git clone <repository-url>
cd customer-churn-prediction

# Run setup script
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate