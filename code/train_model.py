import pandas as pd
import numpy as np
import joblib
import os
import urllib
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Đường dẫn gốc dự án (lên một cấp từ thư mục code), file dữ liệu và thư mục lưu model.
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, "data", "fashion_sales_data.csv")
model_dir = os.path.join(base_dir, "model")
model_filename = os.path.join(model_dir, "model.pkl")

# Kết nối SQL Server (Windows Auth). Lỗi thì engine = None, script vẫn chạy với CSV.
SERVER_NAME = r"Viet"
DATABASE_NAME = "TTCN"

engine = None
try:
    params = urllib.parse.quote_plus(
        f'Driver={{ODBC Driver 17 for SQL Server}};'
        f'Server={SERVER_NAME};'
        f'Database={DATABASE_NAME};'
        f'Trusted_Connection=yes;'
        f'TrustServerCertificate=yes;'
    )
    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
except Exception as e:
    print(f"⚠️ Cảnh báo: Không thể khởi tạo Engine database: {e}")

# Đọc dữ liệu: ưu tiên bảng sales_data; lỗi thì đọc CSV; trống thì thoát.
print("⏳ Đang tải dữ liệu...")
data = pd.DataFrame()

try:
    query = "SELECT * FROM sales_data"
    data = pd.read_sql(query, engine)
    print("✅ Đã tải dữ liệu thành công từ cơ sở dữ liệu SQL Server.")
except Exception as e:
    print("⚠️ Lỗi kết nối CSDL hoặc bảng trống. Tự động chuyển sang đọc từ file CSV...")
    try:
        data = pd.read_csv(data_path)
        print(f"✅ Đã tải dữ liệu dự phòng từ {data_path}.")
    except FileNotFoundError:
        print("❌ File CSV cũng không tồn tại. Hãy chạy create_fashion_data.py trước!")
        exit()

if data.empty:
    print("❌ Dữ liệu trống. Hãy kiểm tra lại.")
    exit()

# Chuẩn hóa ngày, tách tháng/năm; target = sales_vnd; X bỏ cột không dùng; tên cột Title Case cho khớp pipeline/app.
data['sales_date'] = pd.to_datetime(data['sales_date'])
data['Month'] = data['sales_date'].dt.month
data['Year'] = data['sales_date'].dt.year

y = data["sales_vnd"]

X = data.drop(["sales_vnd", "sales_date", "data_id", "uploaded_by", "uploaded_at"], axis=1, errors='ignore')

X.columns = [col.title() for col in X.columns]

# Chuẩn hóa số + one-hot danh mục; gói vào pipeline cùng regressor sau.
numeric_features = [
    "Advertising_Vnd", "Online_Ads_Vnd", "Social_Media_Vnd",
    "Price_Vnd", "Discount_Percent", "Stores_Count",
    "Holiday", "Month", "Year"
]
categorical_features = ["Category", "Weather"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# So sánh vài mô hình; giữ pipeline Random Forest tốt nhất cho đồ án để dump + ghi DB.
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

rf_model = None
rf_score = 0
rf_mse = 0

print("\n🏁 BẮT ĐẦU SO SÁNH MÔ HÌNH:")
print("-" * 60)

for name, model in models.items():
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"{name:<20} | R2: {r2:.4f} | RMSE: {rmse:,.0f} VNĐ")

    if name == "Random Forest":
        rf_model = clf
        rf_score = r2
        rf_mse = mse

print("-" * 60)
print(f"🏆 CHỌN MÔ HÌNH THEO ĐỒ ÁN PDF: Random Forest (R2: {rf_score:.4f})")

# Lưu file .pkl; nếu có DB thì deactivate model cũ và insert bản ghi mới vào models_history.
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

joblib.dump(rf_model, model_filename)
print(f"💾 Đã xuất mô hình thành công tại: {model_filename}")

if engine is not None:
    try:
        with engine.begin() as conn:
            conn.execute(text("UPDATE models_history SET is_active = 0"))
            insert_query = text("""
                INSERT INTO models_history (model_name, mse_score, r2_score, file_path, is_active, trained_by)
                VALUES (:name, :mse, :r2, :path, 1, 1)
            """)
            conn.execute(insert_query, {
                "name": "Random Forest", "mse": float(rf_mse), "r2": float(rf_score), "path": "model.pkl"
            })
        print("✅ Đã cập nhật lịch sử mô hình vào bảng models_history trong Database!")
    except Exception as e:
        print(f"⚠️ Database lỗi (Hoặc chưa chạy database). Bỏ qua bước ghi Log. Lỗi: {e}")
