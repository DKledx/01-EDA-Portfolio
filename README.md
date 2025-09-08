# 📊 EDA Portfolio - Bộ Sưu Tập Phân Tích Dữ Liệu

## 🎯 Mục Tiêu Dự Án

Dự án **EDA Portfolio** là bộ sưu tập các phân tích khám phá dữ liệu (Exploratory Data Analysis) trên nhiều dataset khác nhau. Đây là dự án nền tảng giúp bạn làm quen với việc phân tích dữ liệu, trực quan hóa và hiểu sâu về các kỹ thuật EDA cơ bản.

## 🎓 Kiến Thức Sẽ Học Được

### 📚 Thư Viện Python
- **Pandas**: Xử lý và thao tác dữ liệu
- **NumPy**: Tính toán số học và mảng
- **Matplotlib**: Vẽ biểu đồ cơ bản
- **Seaborn**: Trực quan hóa dữ liệu thống kê
- **Plotly**: Biểu đồ tương tác
- **Jupyter Notebook**: Môi trường phát triển

### 🔍 Kỹ Thuật EDA
- **Data Profiling**: Hiểu cấu trúc dữ liệu
- **Missing Value Analysis**: Phân tích dữ liệu thiếu
- **Outlier Detection**: Phát hiện giá trị ngoại lai
- **Distribution Analysis**: Phân tích phân phối
- **Correlation Analysis**: Phân tích tương quan
- **Feature Engineering**: Kỹ thuật tạo đặc trưng

## 📁 Cấu Trúc Dự Án

```
01-EDA-Portfolio/
├── README.md
├── notebooks/
│   ├── 01-titanic-eda.ipynb
│   ├── 02-iris-eda.ipynb
│   ├── 03-housing-eda.ipynb
│   ├── 03-house-sales-eda.ipynb
│   ├── 04-sales-eda.ipynb
│   └── 05-customer-eda.ipynb
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── src/
│   ├── utils.py
│   ├── visualization.py
│   └── data_processing.py
├── reports/
│   ├── figures/
│   └── insights/
├── requirements.txt
└── .gitignore
```

## 🗂️ Dataset Sẽ Phân Tích

### 1. **Titanic Dataset** 🚢
- **Nguồn**: Kaggle Titanic Competition
- **Mục tiêu**: Phân tích dữ liệu hành khách tàu Titanic
- **Kỹ thuật**: Phân tích survival rate, correlation với các features

### 2. **Iris Dataset** 🌸
- **Nguồn**: sklearn.datasets
- **Mục tiêu**: Phân tích đặc điểm của 3 loài hoa Iris
- **Kỹ thuật**: Phân tích phân phối, scatter plot, box plot

### 3. **Housing Dataset** 🏠
- **Nguồn**: California Housing (sklearn) hoặc Ames Housing (Kaggle)
- **Mục tiêu**: Phân tích dữ liệu bất động sản
- **Kỹ thuật**: Phân tích giá nhà, correlation với location

### 3.1. **House Sales in King County** 🏘️
- **Nguồn**: [Kaggle House Sales in King County](https://www.kaggle.com/datasets/shree1992/housedata)
- **Mục tiêu**: Phân tích dữ liệu bất động sản King County, Washington
- **Kỹ thuật**: Phân tích giá nhà, correlation với features, geographic analysis

### 4. **Sales Dataset** 💰
- **Nguồn**: Kaggle Sales Data hoặc synthetic data
- **Mục tiêu**: Phân tích xu hướng bán hàng
- **Kỹ thuật**: Time series analysis, seasonal patterns

### 5. **Customer Dataset** 👥
- **Nguồn**: Kaggle Customer Analytics hoặc synthetic data
- **Mục tiêu**: Phân tích hành vi khách hàng
- **Kỹ thuật**: Customer segmentation, RFM analysis

## 🚀 Cách Bắt Đầu

### 1. Cài Đặt Môi Trường
```bash
# Tạo virtual environment
python -m venv eda_env
source eda_env/bin/activate  # Linux/Mac
# hoặc
eda_env\Scripts\activate     # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Chạy Jupyter Notebook
```bash
jupyter notebook
```

### 3. Bắt Đầu với Notebook Đầu Tiên
Mở `notebooks/01-titanic-eda.ipynb` và bắt đầu phân tích!

## 📋 Checklist EDA Cho Mỗi Dataset

### ✅ Bước 1: Data Loading & Overview
- [ ] Load dataset
- [ ] Kiểm tra shape, dtypes, memory usage
- [ ] Hiển thị sample data
- [ ] Kiểm tra missing values

### ✅ Bước 2: Data Profiling
- [ ] Statistical summary (describe())
- [ ] Unique values cho categorical features
- [ ] Data distribution analysis

### ✅ Bước 3: Missing Value Analysis
- [ ] Tạo missing value heatmap
- [ ] Phân tích pattern của missing values
- [ ] Đề xuất strategy xử lý

### ✅ Bước 4: Univariate Analysis
- [ ] Histogram cho numerical features
- [ ] Bar chart cho categorical features
- [ ] Box plot để phát hiện outliers

### ✅ Bước 5: Bivariate Analysis
- [ ] Correlation matrix
- [ ] Scatter plots
- [ ] Pair plots
- [ ] Categorical vs numerical analysis

### ✅ Bước 6: Multivariate Analysis
- [ ] Advanced visualizations
- [ ] Feature interactions
- [ ] Pattern discovery

### ✅ Bước 7: Insights & Conclusions
- [ ] Tổng hợp findings
- [ ] Đề xuất next steps
- [ ] Business insights

## 🎨 Visualization Best Practices

### 📊 Biểu Đồ Cơ Bản
- **Histogram**: Phân phối của numerical features
- **Bar Chart**: Frequency của categorical features
- **Box Plot**: Phân phối và outliers
- **Scatter Plot**: Mối quan hệ giữa 2 variables

### 📈 Biểu Đồ Nâng Cao
- **Heatmap**: Correlation matrix
- **Pair Plot**: Tất cả combinations
- **Violin Plot**: Distribution + density
- **Ridge Plot**: Multiple distributions

## 🔧 Tools & Libraries

### Core Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
```

### Additional Tools
- **Pandas Profiling**: Auto EDA reports
- **Sweetviz**: Automated EDA
- **DataPrep**: Data preparation
- **Streamlit**: Interactive dashboards

## 📊 Streamlit Dashboard

Tạo dashboard tương tác để showcase các phân tích:
- Dataset selector
- Interactive plots
- Statistical summaries
- Download reports

## 🎯 Kết Quả Mong Đợi

Sau khi hoàn thành dự án này, bạn sẽ có:

1. **Portfolio EDA**: 5+ notebooks phân tích chi tiết
2. **Visualization Skills**: Thành thạo các loại biểu đồ
3. **Data Intuition**: Khả năng "đọc" dữ liệu
4. **Code Reusability**: Functions tái sử dụng cho EDA
5. **Streamlit App**: Dashboard tương tác

## 📚 Tài Liệu Tham Khảo

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)
- [Kaggle Learn - EDA](https://www.kaggle.com/learn/data-visualization)

## 🏆 Next Steps

Sau khi hoàn thành EDA Portfolio, bạn có thể:
- Áp dụng insights vào các dự án ML tiếp theo
- Tạo automated EDA pipeline
- Phát triển custom visualization tools
- Chuyển sang dự án 2: Iris Classification

---

**Happy Analyzing! 🎉**

*Hãy bắt đầu với notebook đầu tiên và khám phá thế giới dữ liệu thú vị!*
