#!/usr/bin/env python3
"""
Script để tải House Data dataset từ Kaggle
Dataset: House Sales in King County, USA
URL: https://www.kaggle.com/datasets/shree1992/housedata
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def download_house_data():
    """Tải House Data dataset từ Kaggle hoặc tạo synthetic data nếu không có API key"""
    
    # Tạo thư mục data nếu chưa có
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Kiểm tra xem đã có file chưa
    house_file = data_dir / "house_data.csv"
    
    if house_file.exists():
        print("✅ House Data dataset đã tồn tại!")
        return str(house_file)
    
    try:
        # Thử tải từ Kaggle
        import kaggle
        print("📥 Đang tải House Data dataset từ Kaggle...")
        
        # Tải dataset
        kaggle.api.dataset_download_files('shree1992/housedata', path='data/raw', unzip=True)
        
        print("✅ Dataset đã được tải thành công!")
        return str(house_file)
        
    except Exception as e:
        print(f"⚠️ Không thể tải từ Kaggle: {e}")
        print("🔄 Tạo synthetic House Data dataset...")
        
        # Tạo synthetic dataset giống House Data thật
        np.random.seed(42)
        n_houses = 21613  # Số lượng houses trong dataset thật
        
        # Tạo synthetic data dựa trên House Sales in King County dataset
        house_data = create_synthetic_house_data(n_houses)
        house_df = pd.DataFrame(house_data)
        house_df.to_csv(house_file, index=False)
        
        print("✅ Synthetic House Data dataset đã được tạo!")
        return str(house_file)

def create_synthetic_house_data(n_houses):
    """Tạo synthetic data giống House Sales in King County dataset"""
    
    # Features của House Sales in King County dataset
    data = {
        'id': range(1, n_houses + 1),
        'date': pd.date_range('2014-05-02', periods=n_houses, freq='D').strftime('%Y%m%dT000000'),
        'price': np.random.lognormal(mean=13.0, sigma=0.5, size=n_houses),  # House price
        'bedrooms': np.random.poisson(lam=3.4, size=n_houses),  # Number of bedrooms
        'bathrooms': np.random.normal(2.1, 0.8, size=n_houses),  # Number of bathrooms
        'sqft_living': np.random.lognormal(mean=8.0, sigma=0.4, size=n_houses),  # Square feet of living space
        'sqft_lot': np.random.lognormal(mean=9.0, sigma=0.6, size=n_houses),  # Square feet of lot
        'floors': np.random.choice([1, 1.5, 2, 2.5, 3, 3.5], size=n_houses, p=[0.1, 0.2, 0.4, 0.2, 0.08, 0.02]),
        'waterfront': np.random.choice([0, 1], size=n_houses, p=[0.99, 0.01]),  # Waterfront property
        'view': np.random.choice([0, 1, 2, 3, 4], size=n_houses, p=[0.7, 0.15, 0.1, 0.04, 0.01]),  # View rating
        'condition': np.random.choice([1, 2, 3, 4, 5], size=n_houses, p=[0.02, 0.1, 0.2, 0.6, 0.08]),  # Condition rating
        'grade': np.random.choice(range(1, 14), size=n_houses, p=[0.01, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.08, 0.02, 0.01, 0.01]),  # Grade rating
        'sqft_above': np.random.lognormal(mean=7.8, sigma=0.4, size=n_houses),  # Square feet above ground
        'sqft_basement': np.random.lognormal(mean=6.0, sigma=0.8, size=n_houses),  # Square feet basement
        'yr_built': np.random.randint(1900, 2015, size=n_houses),  # Year built
        'yr_renovated': np.random.choice([0] + list(range(1930, 2015)), size=n_houses, p=[0.7] + [0.3/85]*85),  # Year renovated
        'zipcode': np.random.choice([98001, 98002, 98003, 98004, 98005, 98006, 98007, 98008, 98010, 98011, 98014, 98019, 98022, 98023, 98024, 98027, 98028, 98029, 98030, 98031, 98032, 98033, 98034, 98038, 98039, 98040, 98042, 98043, 98045, 98052, 98053, 98055, 98056, 98058, 98059, 98065, 98070, 98072, 98074, 98075, 98077, 98092, 98102, 98103, 98105, 98106, 98107, 98108, 98109, 98112, 98115, 98116, 98117, 98118, 98119, 98122, 98125, 98126, 98133, 98136, 98144, 98146, 98148, 98155, 98166, 98168, 98177, 98178, 98188, 98198, 98199], size=n_houses),
        'lat': np.random.uniform(47.2, 47.8, size=n_houses),  # Latitude
        'long': np.random.uniform(-122.5, -121.3, size=n_houses),  # Longitude
        'sqft_living15': np.random.lognormal(mean=8.0, sigma=0.4, size=n_houses),  # Average sqft of 15 nearest neighbors
        'sqft_lot15': np.random.lognormal(mean=9.0, sigma=0.6, size=n_houses),  # Average lot sqft of 15 nearest neighbors
    }
    
    # Điều chỉnh một số giá trị để có logic hơn
    for i in range(n_houses):
        # sqft_above không được lớn hơn sqft_living
        if data['sqft_above'][i] > data['sqft_living'][i]:
            data['sqft_above'][i] = data['sqft_living'][i] * 0.9
        
        # sqft_basement = sqft_living - sqft_above
        data['sqft_basement'][i] = max(0, data['sqft_living'][i] - data['sqft_above'][i])
        
        # yr_renovated không được sớm hơn yr_built
        if data['yr_renovated'][i] > 0 and data['yr_renovated'][i] < data['yr_built'][i]:
            data['yr_renovated'][i] = 0
        
        # Đảm bảo bedrooms và bathrooms là số nguyên dương
        data['bedrooms'][i] = max(1, int(data['bedrooms'][i]))
        data['bathrooms'][i] = max(0.5, round(data['bathrooms'][i], 1))
        
        # Đảm bảo floors là số hợp lý
        data['floors'][i] = max(1, data['floors'][i])
    
    return data

if __name__ == "__main__":
    file_path = download_house_data()
    print(f"📁 Dataset saved to: {file_path}")
    
    # Load và hiển thị thông tin cơ bản
    df = pd.read_csv(file_path)
    print(f"\n📊 Dataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
