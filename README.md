# 🏠 House Price Prediction App

Ứng dụng AI dự đoán giá nhà sử dụng Python + scikit-learn + Streamlit.

## Cấu trúc thư mục

```
house_price_app/
├── app.py                    # Ứng dụng Streamlit (giao diện)
├── src/
│   ├── generate_data.py      # Tạo dữ liệu giả lập + EDA
│   └── train_model.py        # Huấn luyện mô hình ML
├── data/                     # Dữ liệu và biểu đồ EDA
├── models/                   # Model đã lưu (joblib)
└── requirements.txt
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy theo thứ tự

```bash
# Bước 1: Tạo dữ liệu + EDA
python src/generate_data.py

# Bước 2: Huấn luyện mô hình
python src/train_model.py

# Bước 3: Chạy ứng dụng
streamlit run app.py
```

## Mô hình sử dụng

- **Linear Regression** (baseline)
- **Random Forest Regressor** + GridSearchCV (mô hình chính)

## Đặc trưng đầu vào

| Đặc trưng | Mô tả |
|---|---|
| dien_tich_m2 | Diện tích sàn (m²) |
| so_phong_ngu | Số phòng ngủ |
| so_toilet | Số toilet |
| so_tang | Số tầng |
| tuoi_nha_nam | Tuổi nhà (năm) |
| khoang_cach_trung_tam_km | Khoảng cách đến trung tâm (km) |
| co_gara | Có gara (0/1) |
| mat_tien | Mặt tiền đường (0/1) |

**Đầu ra:** Giá nhà dự đoán (Tỷ VNĐ)
