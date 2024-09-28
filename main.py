from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.neural_network import MLPRegressor
import os

app = FastAPI()

# Đọc dữ liệu từ tệp CSV
df = pd.read_csv("Gia_Vang_2019_2022.csv")
df.columns = ["Date", "Price", "Open", "Vol"]

# Chuyển đổi tất cả các cột thành mảng NumPy
x1 = np.array(df["Open"].values, dtype=float)
x2 = np.array(df["Vol"].values, dtype=float)
y = np.array(df["Price"].values, dtype=float)

# Số lượng quan sát
N = len(y)
alpha = 1.0  # Tham số điều chỉnh
X = np.column_stack((x1, x2))

# Hàm dự đoán giá vàng bằng hồi quy tuyến tính
def predict_gold_price(open_value: float, vol_value: float) -> float:
    m1 = (N * np.sum(x1 * y) - np.sum(x1) * np.sum(y)) / (N * np.sum(x1 ** 2) - (np.sum(x1) ** 2))
    m2 = (N * np.sum(x2 * (y - m1 * x1)) - np.sum(x2) * np.sum(y - m1 * x1)) / (N * np.sum(x2 ** 2) - (np.sum(x2) ** 2))
    b = np.mean(y) - (m1 * np.mean(x1) + m2 * np.mean(x2))
    return m1 * open_value + m2 * vol_value + b

# Các hàm tính MSE và R²
def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def calculate_r2(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

# Dự đoán giá trị y (giá vàng) và tính toán MSE, R² cho hồi quy tuyến tính
y_pred_linear = predict_gold_price(x1, x2)
mse_linear = calculate_mse(y, y_pred_linear)
r2_linear = calculate_r2(y, y_pred_linear)

# Hồi quy Lasso
lasso_model = Lasso(alpha=alpha, max_iter=1000)
lasso_model.fit(X, y)

# Hàm dự đoán cho Lasso
def predict_gold_price_lasso(open_value: float, vol_value: float) -> float:
    return lasso_model.predict(np.array([[open_value, vol_value]]))[0]

# Dự đoán cho Lasso
lasso_y_pred = predict_gold_price_lasso(x1, x2)
mse_lasso = calculate_mse(y, lasso_y_pred)
r2_lasso = calculate_r2(y, lasso_y_pred)

# Hồi quy Ridge
ridge_model = Ridge(alpha=alpha, max_iter=1000)
ridge_model.fit(X, y)

# Hàm dự đoán cho Ridge
def predict_gold_price_ridge(open_value: float, vol_value: float) -> float:
    return ridge_model.predict(np.array([[open_value, vol_value]]))[0]

# Dự đoán cho Ridge
ridge_y_pred = predict_gold_price_ridge(x1, x2)
mse_ridge = calculate_mse(y, ridge_y_pred)
r2_ridge = calculate_r2(y, ridge_y_pred)

# Neural Network Regression với ReLU
neural_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, activation='relu', solver='adam', random_state=0)
neural_model.fit(X, y)

# Hàm dự đoán cho Neural Network
def predict_gold_price_neural(open_value: float, vol_value: float) -> float:
    return neural_model.predict(np.array([[open_value, vol_value]]))[0]

# Dự đoán cho Neural Network
neural_y_pred = predict_gold_price_neural(x1, x2)
mse_neural = calculate_mse(y, neural_y_pred)
r2_neural = calculate_r2(y, neural_y_pred)

# Định nghĩa lớp dữ liệu đầu vào
class PredictionInput(BaseModel):
    open: float
    vol: float

# Endpoint dự đoán giá vàng
@app.post("/predict")
async def predict(input_data: PredictionInput):
    open_value = input_data.open
    vol_value = input_data.vol

    try:
        # Dự đoán giá vàng
        predicted_linear = predict_gold_price(open_value, vol_value)
        predicted_ridge = predict_gold_price_ridge(open_value, vol_value)
        predicted_lasso = predict_gold_price_lasso(open_value, vol_value)
        predicted_neural = predict_gold_price_neural(open_value, vol_value)

        return {
            "predicted_linear": predicted_linear,
            "predicted_ridge": predicted_ridge,
            "predicted_lasso": predicted_lasso,
            "predicted_neural": predicted_neural,
            "mse_linear": mse_linear,
            "r2_linear": r2_linear,
            "mse_ridge": mse_ridge,
            "r2_ridge": r2_ridge,
            "mse_lasso": mse_lasso,
            "r2_lasso": r2_lasso,
            "mse_neural": mse_neural,
            "r2_neural": r2_neural
        }
    except Exception as e:
        return {"error": str(e)}

# Trang chính với form nhập liệu
@app.get("/", response_class=HTMLResponse)
async def get_form():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dự đoán giá vàng</title>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <style>
            body {
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                background-color: #f8f8f8; /* Màu nền nhạt */
            }
            .container {
                max-width: 600px;
                margin: auto;
                background-color: #f0f0f0;
                padding: 20px;
                border-radius: 5px;
                text-align: center; /* Căn giữa nội dung form */
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2 class="card-title">Dự đoán giá vàng</h2>
            <form id="predictionForm">
                <div class="form-group">
                    <label for="open">Giá mở cửa:</label>
                    <input type="number" id="open" name="open" class="form-control" step="any" required>
                </div>
                <div class="form-group">
                    <label for="vol">VoL (K):</label>
                    <input type="number" id="vol" name="vol" class="form-control" step="any" required>
                </div>
                <button type="button" class="btn btn-primary" onclick="predict()">Dự đoán</button>
            </form>
            <div id="result"></div>
        </div>
        <script>
            async function predict() {
                const open = document.getElementById('open').value;
                const vol = document.getElementById('vol').value;

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            open: parseFloat(open),
                            vol: parseFloat(vol)
                        }),
                    });

                    const data = await response.json();
                    const resultDiv = document.getElementById('result');

                    if (data.error) {
                        resultDiv.innerHTML = `<p class="text-danger">Đã xảy ra lỗi: ${data.error}</p>`;
                    } else {
                        resultDiv.innerHTML = `
                            <h4>Kết quả Dự đoán</h4>
                            <ul>
                                <li>Hồi quy tuyến tính: ${data.predicted_linear.toFixed(2)} VNĐ</li>
                                <li>Hồi quy Ridge: ${data.predicted_ridge.toFixed(2)} VNĐ</li>
                                <li>Hồi quy Lasso: ${data.predicted_lasso.toFixed(2)} VNĐ</li>
                                <li>Mạng nơ-ron: ${data.predicted_neural.toFixed(2)} VNĐ</li>
                            </ul>
                            <h5>MSE và R²:</h5>
                            <ul>
                                <li>MSE (Hồi quy tuyến tính): ${data.mse_linear.toFixed(2)}</li>
                                <li>R² (Hồi quy tuyến tính): ${data.r2_linear.toFixed(2)}</li>
                                <li>MSE (Hồi quy Ridge): ${data.mse_ridge.toFixed(2)}</li>
                                <li>R² (Hồi quy Ridge): ${data.r2_ridge.toFixed(2)}</li>
                                <li>MSE (Hồi quy Lasso): ${data.mse_lasso.toFixed(2)}</li>
                                <li>R² (Hồi quy Lasso): ${data.r2_lasso.toFixed(2)}</li>
                                <li>MSE (Mạng nơ-ron): ${data.mse_neural.toFixed(2)}</li>
                                <li>R² (Mạng nơ-ron): ${data.r2_neural.toFixed(2)}</li>
                            </ul>
                        `;
                    }
                } catch (error) {
                    document.getElementById('result').innerHTML = `<p class="text-danger">Đã xảy ra lỗi: ${error.message}</p>`;
                }
            }
        </script>
    </body>
    </html>
    """

# Chạy ứng dụng trên cổng 8000 hoặc cổng được chỉ định từ biến môi trường
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
