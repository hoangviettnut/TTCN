import pandas as pd
import numpy as np
import os
import urllib
import itertools
from sqlalchemy import create_engine, text
from sqlalchemy.types import NVARCHAR

SERVER_NAME = r"Viet"
DATABASE_NAME = "TTCN"

params = urllib.parse.quote_plus(
    f'Driver={{ODBC Driver 17 for SQL Server}};'
    f'Server={SERVER_NAME};'
    f'Database={DATABASE_NAME};'
    f'Trusted_Connection=yes;'
    f'TrustServerCertificate=yes;'
)
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

# Xóa bảng sales_data cũ (nếu có) để tạo lại từ đầu.
print("🧹 Đang dọn dẹp cấu trúc cũ trong database...")
try:
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS sales_data"))
        print("✔️ Đã xóa bảng sales_data cũ!")
except Exception as e:
    try:
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE sales_data"))
            print("✔️ Đã xóa bảng sales_data cũ!")
    except:
        print(f"⚠️ Bảng chưa tồn tại. Bỏ qua bước xóa.")

for folder in ['data', 'model']:
    if not os.path.exists(folder):
        os.makedirs(folder)

np.random.seed(42)

print("🚀 Đang khởi tạo dữ liệu giả lập (Phiên bản GOM NHÓM THEO NGÀY)...")

# Lưới: mỗi cặp (ngày × danh mục) một dòng.
dates = pd.date_range(start="2021-01-01", end="2025-12-31", freq='D')
categories = ["Áo", "Quần", "Váy", "Áo khoác", "Giày"]

grid = list(itertools.product(dates, categories))
db_data = pd.DataFrame(grid, columns=["sales_date", "category"])
db_data["month"] = db_data["sales_date"].dt.month

# Theo từng ngày: thời tiết, ngày lễ, số cửa hàng (chung cho mọi danh mục trong ngày đó).
unique_dates = pd.DataFrame({"sales_date": dates})
unique_dates["month"] = unique_dates["sales_date"].dt.month

def get_weather(month):
    if month in [11, 12, 1, 2]:
        return np.random.choice(["Lạnh", "Mưa", "Nắng"], p=[0.7, 0.1, 0.2])
    elif month in [5, 6, 7, 8]:
        return np.random.choice(["Nóng", "Nắng", "Mưa"], p=[0.4, 0.4, 0.2])
    else:
        return np.random.choice(["Nắng", "Mưa", "Lạnh"], p=[0.6, 0.3, 0.1])

def get_holiday(month):
    if month in [1, 2, 11, 12]:
        return np.random.choice([0, 1], p=[0.85, 0.15])
    else:
        return np.random.choice([0, 1], p=[0.98, 0.02])

unique_dates["weather"] = unique_dates["month"].apply(get_weather)
unique_dates["holiday"] = unique_dates["month"].apply(get_holiday)

unique_dates["stores_count"] = np.random.randint(10, 26, size=len(unique_dates))

db_data = db_data.merge(unique_dates[["sales_date", "weather", "holiday", "stores_count"]], on="sales_date")

# Theo từng dòng (ngày + sản phẩm): giá, ngân sách marketing, khuyến mãi.
n = len(db_data)

price_ranges = {
    "Áo": (150000, 350000),
    "Quần": (250000, 550000),
    "Váy": (200000, 700000),
    "Áo khoác": (450000, 1500000),
    "Giày": (300000, 1200000)
}
db_data["price_vnd"] = db_data["category"].apply(lambda c: np.random.randint(price_ranges[c][0], price_ranges[c][1]))

db_data["advertising_vnd"] = np.random.randint(300, 5000, n) * 1000
db_data["online_ads_vnd"] = np.random.randint(200, 3000, n) * 1000
db_data["social_media_vnd"] = np.random.randint(100, 2000, n) * 1000

db_data["discount_percent"] = np.round(np.random.uniform(0, 0.5, n), 2)

# Doanh số = công thức phi tuyến (marketing, cửa hàng, tương tác category×weather, lễ, giảm giá) + nhiễu nhỏ.
def get_interaction(row):
    c, w = row["category"], row["weather"]
    interaction_map = {
        ("Áo khoác", "Lạnh"): 2.5, ("Áo khoác", "Nóng"): 0.2,
        ("Váy", "Nóng"): 1.5, ("Váy", "Lạnh"): 0.6,
        ("Áo", "Nóng"): 1.3, ("Áo", "Lạnh"): 0.8
    }
    return interaction_map.get((c, w), 1.0)

db_data["interaction"] = db_data.apply(get_interaction, axis=1)

base_marketing = (np.sqrt(db_data["advertising_vnd"]/1000) * 1500 +
                  np.sqrt(db_data["online_ads_vnd"]/1000) * 1200 +
                  np.sqrt(db_data["social_media_vnd"]/1000) * 800)

discount_mult = np.where(db_data["discount_percent"] > 0.2,
                         1 + (db_data["discount_percent"] * 2.5),
                         1 + db_data["discount_percent"])

sales_raw = (base_marketing * db_data["stores_count"] * 15) * db_data["interaction"] * discount_mult * (1 + db_data["holiday"] * 1.8)

noise = np.random.normal(0, 0.10, n)
db_data["sales_vnd"] = sales_raw * (1 + noise)
db_data["sales_vnd"] = np.maximum(db_data["sales_vnd"], 50000).astype(int)

# Chọn cột xuất DB/CSV, gắn uploaded_by mặc định.
final_cols = [
    "sales_date", "category", "advertising_vnd", "online_ads_vnd",
    "social_media_vnd", "price_vnd", "discount_percent", "stores_count",
    "weather", "holiday", "sales_vnd"
]
db_data = db_data[final_cols].copy()
db_data["uploaded_by"] = 1

# Ghi SQL Server + file CSV (thư mục data/ tương đối thư mục chạy script).
print("⏳ Đang đẩy dữ liệu vào Database SQL Server...")
db_data.to_sql(
    'sales_data',
    engine,
    if_exists='append',
    index=False,
    dtype={
        'category': NVARCHAR(50),
        'weather': NVARCHAR(50)
    }
)

db_data.to_csv("data/fashion_sales_data.csv", index=False)
print(f"✅ Hoàn tất! Đã tạo {len(db_data)} dòng. Mỗi ngày trong năm chỉ có ĐÚNG 1 dòng cho 1 loại sản phẩm.")
