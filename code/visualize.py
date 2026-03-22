import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import os

# --- XỬ LÝ ĐƯỜNG DẪN TỰ ĐỘNG ---
# Lấy thư mục gốc (TTCN) bằng cách lùi lại 1 thư mục từ vị trí của file visualize.py
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Định nghĩa các đường dẫn tuyệt đối
data_path = os.path.join(base_dir, "data", "fashion_sales_data.csv")
model_path = os.path.join(base_dir, "model", "model.pkl")
charts_dir = os.path.join(base_dir, "charts")

# Tạo thư mục charts nếu chưa có
if not os.path.exists(charts_dir):
    os.makedirs(charts_dir)

# Cấu hình font
plt.rcParams.update({'font.size': 10})

# 1. Load dữ liệu và Model
print("⏳ Đang tải dữ liệu và mô hình...")
try:
    df = pd.read_csv(data_path)
    df.columns = [col.title() for col in df.columns]
    
    if not os.path.exists(model_path):
        raise FileNotFoundError("Không tìm thấy file model.pkl")
        
    model = joblib.load(model_path)
    print("✅ Đã load data và model: model.pkl")
except FileNotFoundError as e:
    print(f"❌ Lỗi: {e}. Hãy kiểm tra xem file đã tồn tại trong thư mục data/model chưa.")
    exit()

df['Sales_Date'] = pd.to_datetime(df['Sales_Date'])
df['Month'] = df['Sales_Date'].dt.month
df['Year'] = df['Sales_Date'].dt.year

# ---------------------------------------------------------
# BIỂU ĐỒ 1: HEATMAP
# ---------------------------------------------------------
print("📊 Đang vẽ Heatmap...")
plt.figure(figsize=(10, 8))
numeric_cols = ['Sales_Vnd', 'Advertising_Vnd', 'Online_Ads_Vnd', 'Social_Media_Vnd', 
                'Price_Vnd', 'Discount_Percent', 'Stores_Count', 'Holiday']

display_names = ['Sales', 'Advertising', 'Online_Ads', 'Social_Media', 'Price', 'Discount', 'Stores', 'Holiday']
corr_matrix = df[numeric_cols].corr()
corr_matrix.columns = display_names
corr_matrix.index = display_names

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Ma trận tương quan giữa các biến')
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, '1_heatmap.png'))
plt.close()

# ---------------------------------------------------------
# BIỂU ĐỒ 2: DOANH SỐ THEO LOẠI SẢN PHẨM
# ---------------------------------------------------------
print("📊 Đang vẽ Doanh số theo loại sản phẩm...")
plt.figure(figsize=(10, 6))
sales_by_cat = df.groupby('Category')['Sales_Vnd'].sum().sort_values(ascending=False).reset_index()

sns.barplot(x='Sales_Vnd', y='Category', data=sales_by_cat, palette='viridis')
plt.title('Tổng doanh số theo Loại sản phẩm')
plt.xlabel('Tổng Doanh số (VND)')
plt.ylabel('Loại sản phẩm')
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, '1b_sales_by_category.png'))
plt.close()

# ---------------------------------------------------------
# BIỂU ĐỒ 3: FEATURE IMPORTANCE
# ---------------------------------------------------------
print("📊 Đang vẽ Feature Importance...")
regressor = model.named_steps['regressor']
if hasattr(regressor, 'feature_importances_'):
    importances = regressor.feature_importances_
    
    numeric_features = ["Advertising_Vnd", "Online_Ads_Vnd", "Social_Media_Vnd", "Price_Vnd", 
                        "Discount_Percent", "Stores_Count", "Holiday", "Month", "Year"]
    try:
        ohe = model.named_steps['preprocessor'].named_transformers_['cat']
        categorical_features = ["Category", "Weather"]
        ohe_features = ohe.get_feature_names_out(categorical_features)
        feature_names = np.concatenate([numeric_features, ohe_features])
    except Exception as e:
        feature_names = [f"Feature_{i}" for i in range(len(importances))]

    if len(feature_names) == len(importances):
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        fi_df['Feature'] = fi_df['Feature'].str.replace('_Vnd', '').str.replace('_Percent', '').str.replace('_Count', '')
        
        fi_df = fi_df.sort_values(by='Importance', ascending=False).head(10)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=fi_df, hue='Feature', palette='viridis', legend=False)
        plt.title('Top 10 Yếu tố ảnh hưởng đến Doanh số')
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, '2_feature_importance.png'))
        plt.close()

# ---------------------------------------------------------
# BIỂU ĐỒ 4: THỰC TẾ VS DỰ ĐOÁN
# ---------------------------------------------------------
print("📊 Đang vẽ So sánh Thực tế và Dự đoán...")
X = df.drop(["Sales_Vnd", "Sales_Date", "Data_Id", "Uploaded_By", "Uploaded_At"], axis=1, errors='ignore')
y_true = df["Sales_Vnd"] 

y_pred = model.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(y_true, y_pred, alpha=0.4, color='dodgerblue', edgecolor='k', linewidth=0.5)

min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Dự đoán hoàn hảo (y = x)')

plt.xlabel('Doanh thu Thực tế (VNĐ)')
plt.ylabel('Doanh thu Dự đoán (VNĐ)')
plt.title('Biểu đồ phân tán: Doanh thu Thực tế vs. Dự đoán')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, '3_actual_vs_predicted.png'))
plt.close()

print("🎉 Hoàn tất toàn bộ biểu đồ! Bạn có thể xem ảnh trong thư mục 'charts' (nằm ngoài thư mục code).")