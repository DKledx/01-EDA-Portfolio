"""
Utility functions for EDA Portfolio project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load data from various file formats
    
    Args:
        file_path: Path to the data file
        **kwargs: Additional arguments for pandas read functions
    
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path, **kwargs)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        return pd.read_excel(file_path, **kwargs)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def data_overview(df: pd.DataFrame) -> None:
    """
    Display comprehensive data overview
    
    Args:
        df: Input dataframe
    """
    print("=" * 50)
    print("📊 DATA OVERVIEW")
    print("=" * 50)
    
    print(f"Shape: {df.shape}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Columns: {list(df.columns)}")
    
    print("\n📋 Data Types:")
    print(df.dtypes.value_counts())
    
    print("\n🔍 First 5 rows:")
    print(df.head())
    
    print("\n📈 Statistical Summary:")
    print(df.describe(include='all'))

def missing_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze missing values in the dataset
    
    Args:
        df: Input dataframe
    
    Returns:
        pd.DataFrame: Missing value analysis
    """
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100,
        'Data_Type': df.dtypes
    })
    
    missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values(
        'Missing_Percentage', ascending=False
    )
    
    return missing_data

def plot_missing_heatmap(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot missing value heatmap
    
    Args:
        df: Input dataframe
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
    plt.title('Missing Values Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_distributions(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                      figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Plot distributions for numerical columns
    
    Args:
        df: Input dataframe
        columns: List of columns to plot (if None, plot all numerical)
        figsize: Figure size
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    n_cols = 3
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
    
    for i, col in enumerate(columns):
        if i < len(axes):
            df[col].hist(bins=30, ax=axes[i], alpha=0.7, edgecolor='black')
            axes[i].set_title(f'Distribution of {col}', fontweight='bold')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    
    # Hide empty subplots
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_categorical_counts(df: pd.DataFrame, columns: Optional[List[str]] = None,
                          figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Plot value counts for categorical columns
    
    Args:
        df: Input dataframe
        columns: List of columns to plot (if None, plot all categorical)
        figsize: Figure size
    """
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    n_cols = 2
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
    
    for i, col in enumerate(columns):
        if i < len(axes):
            value_counts = df[col].value_counts()
            value_counts.plot(kind='bar', ax=axes[i], color='skyblue', edgecolor='black')
            axes[i].set_title(f'Value Counts of {col}', fontweight='bold')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Count')
            axes[i].tick_params(axis='x', rotation=45)
    
    # Hide empty subplots
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def correlation_analysis(df: pd.DataFrame, target_col: Optional[str] = None,
                        figsize: Tuple[int, int] = (12, 10)) -> pd.DataFrame:
    """
    Perform correlation analysis
    
    Args:
        df: Input dataframe
        target_col: Target column for correlation (if None, show all correlations)
        figsize: Figure size
    
    Returns:
        pd.DataFrame: Correlation matrix
    """
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numerical_df.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # If target column specified, show correlations with target
    if target_col and target_col in corr_matrix.columns:
        target_corr = corr_matrix[target_col].drop(target_col).sort_values(
            key=abs, ascending=False
        )
        print(f"\n🎯 Correlations with {target_col}:")
        print(target_corr)
    
    return corr_matrix

def detect_outliers(df: pd.DataFrame, columns: Optional[List[str]] = None,
                   method: str = 'iqr') -> pd.DataFrame:
    """
    Detect outliers using IQR or Z-score method
    
    Args:
        df: Input dataframe
        columns: List of columns to check (if None, check all numerical)
        method: 'iqr' or 'zscore'
    
    Returns:
        pd.DataFrame: Outlier information
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outlier_info = []
    
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = df[z_scores > 3]
        
        outlier_info.append({
            'Column': col,
            'Outlier_Count': len(outliers),
            'Outlier_Percentage': (len(outliers) / len(df)) * 100
        })
    
    return pd.DataFrame(outlier_info)

def create_summary_report(df: pd.DataFrame, target_col: Optional[str] = None) -> None:
    """
    Create a comprehensive summary report
    
    Args:
        df: Input dataframe
        target_col: Target column name (if applicable)
    """
    print("=" * 60)
    print("📊 COMPREHENSIVE DATA SUMMARY REPORT")
    print("=" * 60)
    
    # Basic info
    print(f"Dataset Shape: {df.shape}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data types
    print(f"\nData Types:")
    print(df.dtypes.value_counts())
    
    # Missing values
    missing_data = missing_analysis(df)
    if not missing_data.empty:
        print(f"\nMissing Values:")
        print(missing_data)
    else:
        print(f"\n✅ No missing values found!")
    
    # Outliers
    outlier_data = detect_outliers(df)
    print(f"\nOutlier Analysis:")
    print(outlier_data)
    
    # Target analysis (if provided)
    if target_col and target_col in df.columns:
        print(f"\n🎯 Target Variable Analysis ({target_col}):")
        if df[target_col].dtype in ['object', 'category']:
            print(df[target_col].value_counts())
        else:
            print(df[target_col].describe())
    
    print("\n" + "=" * 60)
