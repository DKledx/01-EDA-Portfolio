#!/usr/bin/env python3
"""
Test script để kiểm tra notebook và tìm bug
"""

import sys
import os
sys.path.append('src')

# Test imports
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    print("✅ Core libraries imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test custom modules
try:
    from utils import *
    from visualization import *
    print("✅ Custom modules imported successfully!")
except ImportError as e:
    print(f"❌ Custom module import error: {e}")
    sys.exit(1)

# Test matplotlib style
try:
    plt.style.use('seaborn-v0_8')
    print("✅ Seaborn style loaded successfully!")
except OSError:
    try:
        plt.style.use('seaborn')
        print("✅ Fallback seaborn style loaded successfully!")
    except OSError as e:
        print(f"❌ Style error: {e}")

# Test data creation (tương tự như trong notebook)
try:
    np.random.seed(42)
    n_passengers = 891

    # Create synthetic data
    data = {
        'PassengerId': range(1, n_passengers + 1),
        'Survived': np.random.choice([0, 1], n_passengers, p=[0.62, 0.38]),
        'Pclass': np.random.choice([1, 2, 3], n_passengers, p=[0.24, 0.21, 0.55]),
        'Name': [f'Passenger_{i}' for i in range(1, n_passengers + 1)],
        'Sex': np.random.choice(['male', 'female'], n_passengers, p=[0.65, 0.35]),
        'Age': np.random.normal(30, 14, n_passengers).clip(0, 80),
        'SibSp': np.random.poisson(0.5, n_passengers).clip(0, 8),
        'Parch': np.random.poisson(0.4, n_passengers).clip(0, 6),
        'Ticket': [f'Ticket_{i}' for i in range(1, n_passengers + 1)],
        'Fare': np.random.lognormal(2.5, 1.2, n_passengers).clip(0, 512),
        'Cabin': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', None], 
                                 n_passengers, p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3]),
        'Embarked': np.random.choice(['C', 'Q', 'S'], n_passengers, p=[0.19, 0.09, 0.72])
    }

    df = pd.DataFrame(data)
    print("✅ Synthetic data created successfully!")
    print(f"Dataset shape: {df.shape}")
    
except Exception as e:
    print(f"❌ Data creation error: {e}")
    sys.exit(1)

# Test utility functions
try:
    data_overview(df)
    print("✅ data_overview function works!")
except Exception as e:
    print(f"❌ data_overview error: {e}")

try:
    missing_data = missing_analysis(df)
    print("✅ missing_analysis function works!")
    print(f"Missing data shape: {missing_data.shape}")
except Exception as e:
    print(f"❌ missing_analysis error: {e}")

# Test visualization functions
try:
    plotter = EDAPlotter()
    print("✅ EDAPlotter class instantiated successfully!")
except Exception as e:
    print(f"❌ EDAPlotter error: {e}")

try:
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    plotter.plot_numerical_summary(df, numerical_cols[:3])  # Test với 3 columns đầu
    print("✅ plot_numerical_summary function works!")
except Exception as e:
    print(f"❌ plot_numerical_summary error: {e}")

try:
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    plotter.plot_categorical_analysis(df, categorical_cols[:2])  # Test với 2 columns đầu
    print("✅ plot_categorical_analysis function works!")
except Exception as e:
    print(f"❌ plot_categorical_analysis error: {e}")

print("\n🎉 All tests completed! Notebook should work properly now.")
