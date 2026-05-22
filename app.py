"""
Module 3: Ứng dụng Streamlit - Dự đoán giá nhà
Chạy: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Cấu hình trang ─────────────────────────────────────────────
st.set_page_config(
    page_title="Dự Đoán Giá Nhà",
    page_icon="🏠",
    layout="centered"
)

# ── Load mô hình ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "models", "best_model.pkl")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


model = load_model()

# ── Giao diện ──────────────────────────────────────────────────
st.title("🏠 Dự Đoán Giá Nhà")
st.markdown("**Nhập thông tin căn nhà để nhận dự đoán giá (Tỷ VNĐ)**")
st.divider()

col1, col2 = st.columns(2)

with col1:
    dien_tich = st.number_input("Diện tích (m²)", min_value=20, max_value=500, value=80, step=5)
    so_phong_ngu = st.selectbox("Số phòng ngủ", [1, 2, 3, 4, 5], index=1)
    so_toilet = st.selectbox("Số toilet", [1, 2, 3], index=0)
    so_tang = st.selectbox("Số tầng", [1, 2, 3, 4, 5], index=1)

with col2:
    tuoi_nha = st.slider("Tuổi nhà (năm)", 0, 30, 5)
    khoang_cach = st.slider("Khoảng cách trung tâm (km)", 0.5, 20.0, 5.0, step=0.5)
    co_gara = st.checkbox("Có gara", value=False)
    mat_tien = st.checkbox("Mặt tiền đường", value=False)

st.divider()

if st.button("🔍 Dự đoán giá", type="primary", use_container_width=True):
    if model is None:
        st.error("❌ Chưa tìm thấy mô hình. Vui lòng chạy `train_model.py` trước.")
    else:
        input_data = pd.DataFrame([{
            "dien_tich_m2": dien_tich,
            "so_phong_ngu": so_phong_ngu,
            "so_toilet": so_toilet,
            "so_tang": so_tang,
            "tuoi_nha_nam": tuoi_nha,
            "khoang_cach_trung_tam_km": khoang_cach,
            "co_gara": int(co_gara),
            "mat_tien": int(mat_tien)
        }])

        gia_du_doan = model.predict(input_data)[0]
        gia_du_doan = max(0.5, round(gia_du_doan, 2))

        st.success(f"### 💰 Giá dự đoán: **{gia_du_doan:.2f} Tỷ VNĐ**")
        st.caption(
            f"≈ {gia_du_doan * 1000:.0f} Triệu VNĐ  |  "
            f"Đơn giá: {gia_du_doan * 1000 / dien_tich:.1f} Triệu/m²"
        )

        with st.expander("📋 Thông tin đã nhập"):
            st.dataframe(input_data.T.rename(columns={0: "Giá trị"}))

st.divider()
st.caption("⚠️ Kết quả mang tính tham khảo, dựa trên mô hình huấn luyện từ dữ liệu giả lập.")
