import streamlit as st
import pandas as pd
import joblib
import os
import urllib
import datetime
import plotly.express as px
from sqlalchemy import create_engine, text

st.set_page_config(page_title="Hệ thống Dự báo Doanh số AI - TNUT", layout="wide")

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

# Đăng nhập: kiểm tra users; lưu user_id, role, username vào session.
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.title("🔐 Đăng nhập Hệ thống Dự báo")
    with st.form("login_form"):
        u = st.text_input("Tài khoản")
        p = st.text_input("Mật khẩu", type="password")
        submitted = st.form_submit_button("Đăng nhập")

        if submitted:
            with engine.connect() as conn:
                query = text("SELECT user_id, role, username FROM users WHERE username=:u AND password=:p")
                user = conn.execute(query, {"u": u, "p": p}).fetchone()

                if user:
                    st.session_state.update({
                        'logged_in': True,
                        'user_id': user[0],
                        'role': user[1].lower(),
                        'username': user[2]
                    })
                    st.rerun()
                else:
                    st.error("❌ Sai tài khoản hoặc mật khẩu!")
    st.stop()

# Load mô hình đang active từ DB; đường dẫn file nằm cạnh thư mục code/../model/.
base_dir = os.path.dirname(os.path.abspath(__file__))
with engine.connect() as conn:
    row = conn.execute(text("SELECT model_id, file_path, model_name FROM models_history WHERE is_active = 1")).fetchone()

model = None
active_model_info = None

if row:
    filename = os.path.basename(row[1])
    potential_path = os.path.join(base_dir, "..", "model", filename)
    if os.path.exists(potential_path):
        model = joblib.load(potential_path)
        active_model_info = row

# Sidebar: menu theo role, đăng xuất, trạng thái model.
st.sidebar.title(f"👤 {st.session_state['username']}")
st.sidebar.write(f"Quyền hạn: **{st.session_state['role'].upper()}**")

menu = ["Dự đoán đơn lẻ", "Dự báo 12 tháng"]
if st.session_state['role'] == 'admin':
    menu = ["Dự đoán đơn lẻ", "Dự báo 12 tháng", "Nhập dữ liệu (CSV)", "Quản lý mô hình"]

choice = st.sidebar.radio("Chức năng hệ thống", menu)

if st.sidebar.button("Đăng xuất"):
    st.session_state['logged_in'] = False
    st.rerun()

st.sidebar.markdown("---")
if active_model_info:
    st.sidebar.success(f"🤖 Đang dùng: {active_model_info[2]}")
else:
    st.sidebar.error("⚠️ Chưa có mô hình Active!")

# Dự đoán một dòng: form → DataFrame đúng tên cột như lúc train → predict → ghi predictions.
if choice == "Dự đoán đơn lẻ":
    st.header("🔮 Dự báo doanh số theo thời điểm")
    if not model:
        st.warning("Hệ thống chưa có mô hình. Vui lòng liên hệ Admin chạy train_model.py!")
    else:
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                date_input = st.date_input("Ngày/Tháng/Năm", datetime.date(2025, 6, 1))
                category = st.selectbox("Loại sản phẩm", ["Áo", "Quần", "Váy", "Áo khoác", "Giày"])
                weather = st.selectbox("Thời tiết dự kiến", ["Nắng", "Mưa", "Lạnh", "Nóng"])
                holiday = st.checkbox("Ngày lễ/Sự kiện đặc biệt")

            with col2:
                price = st.number_input("Giá niêm yết (VNĐ)", 50000, 5000000, 500000, step=50000)
                discount = st.slider("Khuyến mãi (%)", 0.0, 0.5, 0.1)
                stores = st.number_input("Số lượng cửa hàng", 1, 100, 10)

            with col3:
                advertising = st.number_input("QC Truyền thống (VNĐ)", 0, 10000000, 1500000, step=100000)
                online_ads = st.number_input("QC Online (VNĐ)", 0, 10000000, 1000000, step=100000)
                social_media = st.number_input("Mạng xã hội (VNĐ)", 0, 10000000, 800000, step=100000)

            if st.button("🚀 Thực hiện dự đoán", type="primary"):
                input_df = pd.DataFrame([{
                    "Category": category,
                    "Advertising_Vnd": advertising,
                    "Online_Ads_Vnd": online_ads,
                    "Social_Media_Vnd": social_media,
                    "Price_Vnd": price,
                    "Discount_Percent": discount,
                    "Stores_Count": stores,
                    "Weather": weather,
                    "Holiday": 1 if holiday else 0,
                    "Month": date_input.month,
                    "Year": date_input.year
                }])

                prediction = int(model.predict(input_df)[0])
                st.success(f"💰 Doanh số dự kiến cho ngày {date_input.strftime('%d/%m/%Y')}: **{prediction:,} VNĐ**")

                with engine.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO predictions (user_id, model_id, input_date, input_category, predicted_sales)
                        VALUES (:uid, :mid, :idate, :icat, :psales)
                    """), {
                        "uid": st.session_state['user_id'], "mid": active_model_info[0],
                        "idate": date_input, "icat": category, "psales": prediction
                    })

# 12 tháng: có dữ liệu thật → so sánh actual vs predict (admin); không có → kịch bản mặc định + chỉ đường dự báo.
elif choice == "Dự báo 12 tháng":
    st.header("📊 Phân tích xu hướng 12 tháng")

    col_a, col_b = st.columns(2)
    with col_a:
        sel_cat = st.selectbox("Chọn sản phẩm phân tích", ["Áo", "Quần", "Váy", "Áo khoác", "Giày"])
    with col_b:
        sel_year = st.number_input("Năm dự báo", 2024, 2030, 2025)

    if st.button("📈 Lập biểu đồ dự báo"):
        if not model:
            st.error("Chưa tải được mô hình!")
            st.stop()

        query = f"SELECT * FROM sales_data WHERE YEAR(sales_date) = {sel_year} AND category = N'{sel_cat}'"
        df_db = pd.read_sql(query, engine)

        if not df_db.empty:
            st.info(f"Đã tìm thấy dữ liệu gốc năm {sel_year}. Hệ thống đang chạy đối chiếu AI...")

            df_db['sales_date'] = pd.to_datetime(df_db['sales_date'])
            df_db['Month'] = df_db['sales_date'].dt.month
            df_db['Year'] = df_db['sales_date'].dt.year

            X = df_db.drop(["sales_vnd", "sales_date", "data_id", "uploaded_by", "uploaded_at"], axis=1, errors='ignore')
            X.columns = [col.title() for col in X.columns]

            df_db['Predicted_Sales'] = model.predict(X)
            df_db['Actual_Sales'] = df_db['sales_vnd']

            df_monthly = df_db.groupby('Month')[['Actual_Sales', 'Predicted_Sales']].sum().reset_index()

            if st.session_state['role'] == 'admin':
                st.success("👑 Đặc quyền Admin: Hiển thị biểu đồ đối chiếu Thực tế vs Dự báo AI.")
                df_melt = df_monthly.melt(id_vars='Month', value_vars=['Actual_Sales', 'Predicted_Sales'],
                                          var_name='Nguồn', value_name='Doanh số (VNĐ)')
                df_melt['Nguồn'] = df_melt['Nguồn'].replace({'Actual_Sales': 'Thực tế', 'Predicted_Sales': 'AI Dự báo'})

                fig = px.line(df_melt, x='Month', y='Doanh số (VNĐ)', color='Nguồn',
                              title=f"KIỂM CHỨNG: Doanh số Thực tế vs AI - {sel_cat} ({sel_year})",
                              markers=True, template="plotly_dark",
                              color_discrete_map={"Thực tế": "#00FF00", "AI Dự báo": "#FF3366"})

                fig.update_layout(xaxis=dict(tickmode='linear', tick0=1, dtick=1))
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df_monthly.rename(columns={"Actual_Sales": "Thực Tế (VNĐ)", "Predicted_Sales": "AI Dự Báo (VNĐ)"}).set_index('Month'))
            else:
                fig = px.line(df_monthly, x='Month', y='Predicted_Sales',
                              title=f"Xu hướng dự báo doanh số - {sel_cat} năm {sel_year}",
                              markers=True, template="plotly_dark")
                fig.update_layout(xaxis=dict(tickmode='linear', tick0=1, dtick=1), yaxis_title="Doanh số (VNĐ)")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df_monthly[['Month', 'Predicted_Sales']].set_index('Month').rename(columns={"Predicted_Sales": "AI Dự Báo (VNĐ)"}))

        else:
            st.info(f"Chưa có dữ liệu gốc năm {sel_year}. Hệ thống tự động tạo kịch bản giả định để dự báo.")

            forecast_list = []
            for m in range(1, 13):
                wea = "Lạnh" if m in [1, 2, 12] else "Nóng" if m in [5, 6, 7] else "Nắng"
                forecast_list.append({
                    "Category": sel_cat,
                    "Advertising_Vnd": 1500000,
                    "Online_Ads_Vnd": 1000000,
                    "Social_Media_Vnd": 800000,
                    "Price_Vnd": 500000,
                    "Discount_Percent": 0.1,
                    "Stores_Count": 15,
                    "Weather": wea,
                    "Holiday": 1 if m in [1, 2, 5, 9] else 0,
                    "Month": m,
                    "Year": sel_year
                })

            df_forecast = pd.DataFrame(forecast_list)
            df_forecast['Predicted_Sales'] = model.predict(df_forecast)

            fig = px.line(df_forecast, x='Month', y='Predicted_Sales',
                          title=f"Kịch bản dự báo doanh số {sel_cat} năm {sel_year}",
                          markers=True, line_shape="spline", template="plotly_dark")
            fig.update_layout(xaxis=dict(tickmode='linear', tick0=1, dtick=1), yaxis_title="Doanh số (VNĐ)")

            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_forecast[['Month', 'Predicted_Sales']].set_index('Month').rename(columns={"Predicted_Sales": "AI Dự Báo (VNĐ)"}))

# Admin: upload CSV → chuẩn hóa tên cột → append vào sales_data.
elif choice == "Nhập dữ liệu (CSV)":
    st.header("📂 Quản lý dữ liệu bán hàng")
    st.markdown("Dành cho quản trị viên cập nhật dữ liệu huấn luyện mới.")

    file = st.file_uploader("Tải lên file CSV dữ liệu bán hàng thực tế", type="csv")
    if file:
        data_new = pd.read_csv(file)
        st.write("Xem trước dữ liệu:", data_new.head())
        if st.button("📥 Đẩy vào CSDL SQL Server"):
            try:
                data_new.columns = [c.lower() for c in data_new.columns]
                data_new['uploaded_by'] = st.session_state['user_id']
                data_new.to_sql('sales_data', engine, if_exists='append', index=False)
                st.success("✅ Đã cập nhật dữ liệu thành công!")
            except Exception as e:
                st.error(f"Lỗi: {e}")

# Admin: xem lịch sử models_history.
elif choice == "Quản lý mô hình":
    st.header("📈 Đánh giá hiệu suất AI")
    with engine.connect() as conn:
        history = pd.read_sql("SELECT model_name, r2_score, mse_score, trained_at, is_active FROM models_history ORDER BY trained_at DESC", conn)

    st.table(history)
    st.info("💡 Mẹo: R2 Score càng gần 1.0 mô hình càng chính xác. MSE càng nhỏ sai số càng ít.")
