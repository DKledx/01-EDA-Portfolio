#!/usr/bin/env python3
"""
Script để tải dataset Titanic từ Kaggle
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def download_titanic_data():
    """Tải dataset Titanic từ Kaggle hoặc tạo synthetic data nếu không có API key"""
    
    # Tạo thư mục data nếu chưa có
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Kiểm tra xem đã có file chưa
    train_file = data_dir / "train.csv"
    test_file = data_dir / "test.csv"
    
    if train_file.exists() and test_file.exists():
        print("✅ Dataset đã tồn tại!")
        return str(train_file), str(test_file)
    
    try:
        # Thử tải từ Kaggle
        import kaggle
        print("📥 Đang tải dataset Titanic từ Kaggle...")
        
        # Tải dataset
        kaggle.api.dataset_download_files('c/titanic', path='data/raw', unzip=True)
        
        print("✅ Dataset đã được tải thành công!")
        return str(train_file), str(test_file)
        
    except Exception as e:
        print(f"⚠️ Không thể tải từ Kaggle: {e}")
        print("🔄 Tạo synthetic dataset...")
        
        # Tạo synthetic dataset giống thật
        np.random.seed(42)
        n_train = 891
        n_test = 418
        
        # Tạo training data
        train_data = create_synthetic_titanic_data(n_train, is_train=True)
        train_df = pd.DataFrame(train_data)
        train_df.to_csv(train_file, index=False)
        
        # Tạo test data
        test_data = create_synthetic_titanic_data(n_test, is_train=False)
        test_df = pd.DataFrame(test_data)
        test_df.to_csv(test_file, index=False)
        
        print("✅ Synthetic dataset đã được tạo!")
        return str(train_file), str(test_file)

def create_synthetic_titanic_data(n_passengers, is_train=True):
    """Tạo synthetic data giống dataset Titanic thật"""
    
    # Tạo dữ liệu cơ bản
    data = {
        'PassengerId': range(1, n_passengers + 1),
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
    
    # Thêm missing values thực tế
    # Age missing ~20%
    age_missing = np.random.choice([True, False], n_passengers, p=[0.2, 0.8])
    data['Age'] = [None if missing else age for missing, age in zip(age_missing, data['Age'])]
    
    # Cabin missing ~77%
    cabin_missing = np.random.choice([True, False], n_passengers, p=[0.77, 0.23])
    data['Cabin'] = [None if missing else cabin for missing, cabin in zip(cabin_missing, data['Cabin'])]
    
    # Embarked missing ~0.2%
    embarked_missing = np.random.choice([True, False], n_passengers, p=[0.002, 0.998])
    data['Embarked'] = [None if missing else embarked for missing, embarked in zip(embarked_missing, data['Embarked'])]
    
    # Thêm Survived cho training data
    if is_train:
        # Tạo realistic survival patterns
        survived = np.zeros(n_passengers)
        
        # Women more likely to survive
        female_mask = np.array(data['Sex']) == 'female'
        survived[female_mask] = np.random.choice([0, 1], np.sum(female_mask), p=[0.26, 0.74])
        
        # Men less likely to survive
        male_mask = np.array(data['Sex']) == 'male'
        survived[male_mask] = np.random.choice([0, 1], np.sum(male_mask), p=[0.81, 0.19])
        
        # Higher class more likely to survive
        for pclass in [1, 2, 3]:
            class_mask = np.array(data['Pclass']) == pclass
            if pclass == 1:
                survived[class_mask] = np.random.choice([0, 1], np.sum(class_mask), p=[0.37, 0.63])
            elif pclass == 2:
                survived[class_mask] = np.random.choice([0, 1], np.sum(class_mask), p=[0.53, 0.47])
            else:  # pclass == 3
                survived[class_mask] = np.random.choice([0, 1], np.sum(class_mask), p=[0.76, 0.24])
        
        data['Survived'] = survived.astype(int)
    
    return data

if __name__ == "__main__":
    train_file, test_file = download_titanic_data()
    print(f"Train file: {train_file}")
    print(f"Test file: {test_file}")
    
    # Kiểm tra data
    train_df = pd.read_csv(train_file)
    print(f"\nTrain dataset shape: {train_df.shape}")
    print(f"Columns: {list(train_df.columns)}")
    print(f"\nFirst few rows:")
    print(train_df.head())
