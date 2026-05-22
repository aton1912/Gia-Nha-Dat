"""
Module 1: Tạo dữ liệu giả lập và thực hiện EDA
Bài toán: Dự đoán giá nhà dựa trên các đặc trưng
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

np.random.seed(42)

def generate_dataset(n=300):
    """Tạo dataset giả lập về giá nhà"""
    dien_tich = np.random.randint(40, 200, n)
    so_phong_ngu = np.random.randint(1, 6, n)
    so_toilet = np.random.randint(1, 4, n)
    so_tang = np.random.randint(1, 5, n)
    tuoi_nha = np.random.randint(0, 30, n)
    khoang_cach_trung_tam = np.round(np.random.uniform(0.5, 20.0, n), 1)
    co_gara = np.random.randint(0, 2, n)
    mat_tien = np.random.randint(0, 2, n)

    # Công thức giá (tỷ VND) có noise
    gia_co_ban = (
        dien_tich * 0.05
        + so_phong_ngu * 0.3
        + so_toilet * 0.2
        + so_tang * 0.15
        - tuoi_nha * 0.02
        - khoang_cach_trung_tam * 0.1
        + co_gara * 0.5
        + mat_tien * 0.8
        + 1.0
    )
    noise = np.random.normal(0, 0.3, n)
    gia_nha = np.round(np.clip(gia_co_ban + noise, 0.5, 20.0), 2)

    df = pd.DataFrame({
        "dien_tich_m2": dien_tich,
        "so_phong_ngu": so_phong_ngu,
        "so_toilet": so_toilet,
        "so_tang": so_tang,
        "tuoi_nha_nam": tuoi_nha,
        "khoang_cach_trung_tam_km": khoang_cach_trung_tam,
        "co_gara": co_gara,
        "mat_tien": mat_tien,
        "gia_ty_vnd": gia_nha
    })
    return df


def perform_eda(df, output_dir="data"):
    """Thực hiện Exploratory Data Analysis"""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 50)
    print("THÔNG TIN DATASET")
    print("=" * 50)
    print(f"Số dòng: {df.shape[0]}, Số cột: {df.shape[1]}")
    print("\nKiểu dữ liệu:")
    print(df.dtypes)
    print("\nThống kê mô tả:")
    print(df.describe().round(2))
    print(f"\nGiá trị thiếu:\n{df.isnull().sum()}")

    # Hình 1: Phân phối biến mục tiêu
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(df["gia_ty_vnd"], bins=30, color="#4C72B0", edgecolor="white")
    plt.title("Phân phối Giá nhà (Tỷ VNĐ)")
    plt.xlabel("Giá (Tỷ VNĐ)")
    plt.ylabel("Tần số")

    plt.subplot(1, 2, 2)
    sns.boxplot(y=df["gia_ty_vnd"], color="#4C72B0")
    plt.title("Boxplot Giá nhà")
    plt.ylabel("Giá (Tỷ VNĐ)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/eda_phan_phoi_gia.png", dpi=100, bbox_inches="tight")
    plt.close()

    # Hình 2: Tương quan
    plt.figure(figsize=(8, 6))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Ma trận tương quan")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/eda_tuong_quan.png", dpi=100, bbox_inches="tight")
    plt.close()

    # Hình 3: Diện tích vs Giá
    plt.figure(figsize=(6, 4))
    plt.scatter(df["dien_tich_m2"], df["gia_ty_vnd"], alpha=0.5, color="#4C72B0")
    plt.xlabel("Diện tích (m²)")
    plt.ylabel("Giá (Tỷ VNĐ)")
    plt.title("Diện tích vs Giá nhà")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/eda_dien_tich_vs_gia.png", dpi=100, bbox_inches="tight")
    plt.close()

    print("\nEDA hoàn thành. Biểu đồ đã lưu vào thư mục data/")
    return df


if __name__ == "__main__":
    df = generate_dataset(300)
    df.to_csv("data/house_data.csv", index=False)
    print("Dataset đã lưu: data/house_data.csv")
    perform_eda(df, output_dir="data")
