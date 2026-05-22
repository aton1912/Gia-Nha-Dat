"""
Module 2: Huấn luyện và đánh giá mô hình học máy
Mô hình: Random Forest Regressor + Linear Regression
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


def load_data(filepath="data/house_data.csv"):
    df = pd.read_csv(filepath)
    X = df.drop(columns=["gia_ty_vnd"])
    y = df["gia_ty_vnd"]
    return X, y


def train_and_evaluate(X, y, output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    # --- Mô hình 1: Linear Regression ---
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)
    results["LinearRegression"] = {
        "MAE": mean_absolute_error(y_test, y_pred_lr),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        "R2": r2_score(y_test, y_pred_lr),
        "model": lr_pipeline,
        "y_pred": y_pred_lr
    }

    # --- Mô hình 2: Random Forest ---
    rf_params = {
        "model__n_estimators": [50, 100],
        "model__max_depth": [None, 10]
    }
    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(random_state=42))
    ])
    grid_search = GridSearchCV(rf_pipeline, rf_params, cv=3, scoring="r2", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    y_pred_rf = best_rf.predict(X_test)
    results["RandomForest"] = {
        "MAE": mean_absolute_error(y_test, y_pred_rf),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        "R2": r2_score(y_test, y_pred_rf),
        "model": best_rf,
        "y_pred": y_pred_rf,
        "best_params": grid_search.best_params_
    }

    # In kết quả
    print("=" * 55)
    print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 55)
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  MAE  = {res['MAE']:.4f} tỷ VNĐ")
        print(f"  RMSE = {res['RMSE']:.4f} tỷ VNĐ")
        print(f"  R²   = {res['R2']:.4f}")
        if "best_params" in res:
            print(f"  Best params: {res['best_params']}")

    # Lưu mô hình tốt nhất (Random Forest)
    joblib.dump(best_rf, f"{output_dir}/best_model.pkl")
    joblib.dump(list(X.columns), f"{output_dir}/feature_names.pkl")
    print(f"\nMô hình tốt nhất đã lưu: {output_dir}/best_model.pkl")

    # Biểu đồ so sánh thực tế vs dự đoán
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, res) in zip(axes, results.items()):
        ax.scatter(y_test, res["y_pred"], alpha=0.5, color="#4C72B0", s=30)
        mn, mx = y_test.min(), y_test.max()
        ax.plot([mn, mx], [mn, mx], "r--", lw=1.5)
        ax.set_xlabel("Giá thực tế (Tỷ VNĐ)")
        ax.set_ylabel("Giá dự đoán (Tỷ VNĐ)")
        ax.set_title(f"{name}\nR² = {res['R2']:.4f}")
    plt.tight_layout()
    plt.savefig("data/model_comparison.png", dpi=100, bbox_inches="tight")
    plt.close()

    # Feature importance (Random Forest)
    rf_model = best_rf.named_steps["model"]
    importances = pd.Series(
        rf_model.feature_importances_, index=X.columns
    ).sort_values(ascending=True)
    plt.figure(figsize=(7, 5))
    importances.plot(kind="barh", color="#4C72B0")
    plt.title("Mức độ quan trọng của các đặc trưng (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("data/feature_importance.png", dpi=100, bbox_inches="tight")
    plt.close()

    return results, X_test, y_test


if __name__ == "__main__":
    X, y = load_data()
    train_and_evaluate(X, y)
