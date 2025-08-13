"""
Visualization functions for churn analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go
import plotly.express as px

# Set style
plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")

def plot_churn_distribution(df, save_path=None):
    """Plot churn distribution"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    churn_counts = df['Churn'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    
    ax.pie(churn_counts.values, labels=['No Churn', 'Churn'], 
           autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('Customer Churn Distribution', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_distribution(df, feature, target='Churn', save_path=None):
    """Plot feature distribution by churn status"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Distribution plot
    for churn_val in df[target].unique():
        subset = df[df[target] == churn_val][feature]
        ax1.hist(subset, alpha=0.7, label=f'{target}={churn_val}', bins=30)
    
    ax1.set_xlabel(feature)
    ax1.set_ylabel('Count')
    ax1.set_title(f'{feature} Distribution by {target}')
    ax1.legend()
    
    # Box plot
    df.boxplot(column=feature, by=target, ax=ax2)
    ax2.set_title(f'{feature} by {target}')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_correlation_heatmap(df, save_path=None):
    """Plot correlation heatmap for numerical features"""
    # Select numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[num_cols].corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, vmin=-1, vmax=1,
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curves(models_dict, X_test, y_test, save_path=None):
    """Plot ROC curves for multiple models"""
    plt.figure(figsize=(10, 8))
    
    for name, model in models_dict.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, linewidth=2, 
                label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_importance(feature_importance_df, top_n=15, save_path=None):
    """Plot top feature importances"""
    top_features = feature_importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_interactive_churn_analysis(df):
    """Create interactive Plotly visualizations"""
    # Churn by contract type
    fig1 = px.histogram(df, x='Contract', color='Churn', 
                       title='Churn Rate by Contract Type',
                       barmode='group')
    
    # Monthly charges distribution
    fig2 = px.box(df, x='Churn', y='MonthlyCharges', 
                  title='Monthly Charges Distribution by Churn Status')
    
    # Tenure vs Total Charges
    fig3 = px.scatter(df, x='tenure', y='TotalCharges', color='Churn',
                     title='Tenure vs Total Charges by Churn Status',
                     hover_data=['MonthlyCharges'])
    
    return fig1, fig2, fig3

def create_churn_plots(df, output_dir='../5_reports/figures/'):
    """Create all standard plots for churn analysis"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Churn distribution
    plot_churn_distribution(df, save_path=os.path.join(output_dir, 'churn_distribution.png'))
    
    # Feature distributions
    for feature in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        if feature in df.columns:
            plot_feature_distribution(df, feature, save_path=os.path.join(output_dir, f'{feature}_distribution.png'))
    
    # Correlation heatmap
    plot_correlation_heatmap(df, save_path=os.path.join(output_dir, 'correlation_heatmap.png'))
    
    return output_dir

if __name__ == "__main__":
    # Test visualizations
    import os
    from .utils import PROCESSED_DATA_PATH
    
    df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'churn_data_processed.csv'))
    create_churn_plots(df)