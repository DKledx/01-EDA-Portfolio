"""
Advanced visualization functions for EDA Portfolio project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Set style
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    plt.style.use('seaborn')
sns.set_palette("husl")

class EDAPlotter:
    """Advanced plotting class for EDA"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    def plot_numerical_summary(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
        """
        Create comprehensive numerical summary plots
        
        Args:
            df: Input dataframe
            columns: List of numerical columns to plot
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        n_cols = 2
        n_rows = (len(columns) + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, col in enumerate(columns):
            if i < len(axes):
                # Create subplot with histogram and box plot
                ax1 = axes[i]
                ax2 = ax1.twinx()
                
                # Histogram
                ax1.hist(df[col].dropna(), bins=30, alpha=0.7, color=self.colors[0], edgecolor='black')
                ax1.set_xlabel(col)
                ax1.set_ylabel('Frequency', color=self.colors[0])
                ax1.tick_params(axis='y', labelcolor=self.colors[0])
                
                # Box plot
                box_data = ax2.boxplot(df[col].dropna(), patch_artist=True, 
                                     boxprops=dict(facecolor=self.colors[1], alpha=0.7))
                ax2.set_ylabel('Box Plot', color=self.colors[1])
                ax2.tick_params(axis='y', labelcolor=self.colors[1])
                
                ax1.set_title(f'{col} - Distribution & Outliers', fontweight='bold', fontsize=12)
        
        # Hide empty subplots
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_categorical_analysis(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
        """
        Create comprehensive categorical analysis plots
        
        Args:
            df: Input dataframe
            columns: List of categorical columns to plot
        """
        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in columns:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Value counts bar plot
            value_counts = df[col].value_counts()
            value_counts.plot(kind='bar', ax=axes[0], color=self.colors[0], edgecolor='black')
            axes[0].set_title(f'{col} - Value Counts', fontweight='bold')
            axes[0].set_xlabel(col)
            axes[0].set_ylabel('Count')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Pie chart
            value_counts.plot(kind='pie', ax=axes[1], autopct='%1.1f%%', 
                            colors=self.colors[:len(value_counts)])
            axes[1].set_title(f'{col} - Distribution', fontweight='bold')
            axes[1].set_ylabel('')
            
            plt.tight_layout()
            plt.show()
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, target_col: Optional[str] = None,
                               figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Create advanced correlation heatmap
        
        Args:
            df: Input dataframe
            target_col: Target column for focused analysis
            figsize: Figure size
        """
        numerical_df = df.select_dtypes(include=[np.number])
        corr_matrix = numerical_df.corr()
        
        plt.figure(figsize=figsize)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   fmt='.2f', annot_kws={'size': 10})
        
        plt.title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        
        # If target column specified, create focused correlation plot
        if target_col and target_col in corr_matrix.columns:
            target_corr = corr_matrix[target_col].drop(target_col).sort_values(key=abs, ascending=True)
            
            plt.figure(figsize=(10, 8))
            colors = ['red' if x < 0 else 'blue' for x in target_corr.values]
            bars = plt.barh(range(len(target_corr)), target_corr.values, color=colors, alpha=0.7)
            plt.yticks(range(len(target_corr)), target_corr.index)
            plt.xlabel('Correlation with ' + target_col)
            plt.title(f'Feature Correlations with {target_col}', fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, target_corr.values)):
                plt.text(value + (0.01 if value > 0 else -0.01), i, f'{value:.3f}', 
                        va='center', ha='left' if value > 0 else 'right')
            
            plt.tight_layout()
            plt.show()
    
    def plot_pairplot_advanced(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                             target_col: Optional[str] = None, sample_size: int = 1000) -> None:
        """
        Create advanced pairplot with sampling for large datasets
        
        Args:
            df: Input dataframe
            columns: List of columns to include
            target_col: Target column for coloring
            sample_size: Sample size for large datasets
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Sample data if too large
        plot_df = df[columns].copy()
        if len(plot_df) > sample_size:
            plot_df = plot_df.sample(n=sample_size, random_state=42)
            print(f"📊 Sampled {sample_size} rows for pairplot visualization")
        
        # Add target column if specified
        if target_col and target_col in df.columns:
            plot_df[target_col] = df[target_col].iloc[plot_df.index]
            hue = target_col
        else:
            hue = None
        
        # Create pairplot
        g = sns.pairplot(plot_df, hue=hue, diag_kind='hist', 
                        plot_kws={'alpha': 0.6, 's': 50},
                        diag_kws={'alpha': 0.7, 'bins': 20})
        
        g.fig.suptitle('Advanced Pairplot Analysis', y=1.02, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_time_series(self, df: pd.DataFrame, date_col: str, value_cols: List[str],
                        figsize: Tuple[int, int] = (15, 8)) -> None:
        """
        Create time series plots
        
        Args:
            df: Input dataframe
            date_col: Date column name
            value_cols: List of value columns to plot
            figsize: Figure size
        """
        df_ts = df.copy()
        df_ts[date_col] = pd.to_datetime(df_ts[date_col])
        df_ts = df_ts.sort_values(date_col)
        
        fig, axes = plt.subplots(len(value_cols), 1, figsize=figsize, sharex=True)
        if len(value_cols) == 1:
            axes = [axes]
        
        for i, col in enumerate(value_cols):
            axes[i].plot(df_ts[date_col], df_ts[col], color=self.colors[i % len(self.colors)], 
                        linewidth=2, alpha=0.8)
            axes[i].set_title(f'{col} Over Time', fontweight='bold')
            axes[i].set_ylabel(col)
            axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel(date_col)
        plt.suptitle('Time Series Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_interactive_dashboard(self, df: pd.DataFrame, target_col: Optional[str] = None) -> None:
        """
        Create interactive Plotly dashboard
        
        Args:
            df: Input dataframe
            target_col: Target column for analysis
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Distribution', 'Correlation', 'Box Plot', 'Scatter'),
            specs=[[{"type": "histogram"}, {"type": "heatmap"}],
                   [{"type": "box"}, {"type": "scatter"}]]
        )
        
        # Distribution plot
        if numerical_cols:
            fig.add_trace(
                go.Histogram(x=df[numerical_cols[0]], name=numerical_cols[0]),
                row=1, col=1
            )
        
        # Correlation heatmap
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr()
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                          colorscale='RdBu', zmid=0),
                row=1, col=2
            )
        
        # Box plot
        if numerical_cols:
            for col in numerical_cols[:3]:  # Limit to 3 columns
                fig.add_trace(
                    go.Box(y=df[col], name=col),
                    row=2, col=1
                )
        
        # Scatter plot
        if len(numerical_cols) >= 2:
            color_col = target_col if target_col in df.columns else None
            fig.add_trace(
                go.Scatter(x=df[numerical_cols[0]], y=df[numerical_cols[1]],
                          mode='markers', name='Scatter',
                          marker=dict(color=df[color_col] if color_col else None,
                                    colorscale='Viridis', showscale=bool(color_col))),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=False, 
                         title_text="Interactive EDA Dashboard")
        fig.show()

def create_eda_report(df: pd.DataFrame, target_col: Optional[str] = None,
                     save_path: Optional[str] = None) -> None:
    """
    Create comprehensive EDA report with all visualizations
    
    Args:
        df: Input dataframe
        target_col: Target column name
        save_path: Path to save the report
    """
    plotter = EDAPlotter()
    
    print("🎨 Creating Comprehensive EDA Report...")
    print("=" * 50)
    
    # 1. Data Overview
    print("1. Data Overview")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data Types:\n{df.dtypes.value_counts()}")
    
    # 2. Missing Values
    print("\n2. Missing Values Analysis")
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        print(missing_data[missing_data > 0])
    else:
        print("✅ No missing values!")
    
    # 3. Numerical Analysis
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numerical_cols:
        print(f"\n3. Numerical Analysis ({len(numerical_cols)} columns)")
        plotter.plot_numerical_summary(df, numerical_cols)
    
    # 4. Categorical Analysis
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        print(f"\n4. Categorical Analysis ({len(categorical_cols)} columns)")
        plotter.plot_categorical_analysis(df, categorical_cols)
    
    # 5. Correlation Analysis
    if len(numerical_cols) > 1:
        print("\n5. Correlation Analysis")
        plotter.plot_correlation_heatmap(df, target_col)
    
    # 6. Pairplot
    if len(numerical_cols) > 1:
        print("\n6. Pairplot Analysis")
        plotter.plot_pairplot_advanced(df, numerical_cols, target_col)
    
    # 7. Interactive Dashboard
    print("\n7. Interactive Dashboard")
    plotter.plot_interactive_dashboard(df, target_col)
    
    print("\n✅ EDA Report Complete!")
    print("=" * 50)
