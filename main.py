import os
from fastapi import FastAPI
import uvicorn
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neural_network import MLPRegressor
import uvicorn
import os

app = FastAPI()

# Đọc dữ liệu từ tệp CSV
df = pd.read_csv("Gia_Vang_2019_2022.csv")
df.columns = ["Date", "Price", "Open", "Vol"]

# Chuyển đổi tất cả các cột thành mảng NumPy
x1 = np.array(df["Open"].values, dtype=float).reshape(-1, 1)  # Chuyển đổi thành ma trận cột
x2 = np.array(df["Vol"].values, dtype=float).reshape(-1, 1)   # Chuyển đổi thành ma trận cột
y = np.array(df["Price"].values, dtype=float)

# Số lượng quan sát
N = len(y)
X = np.hstack((x1, x2))  # Kết hợp x1 và x2 thành một ma trận

# Hồi quy tuyến tính
linear_model = LinearRegression()
linear_model.fit(X, y)

# Hồi quy Lasso
lasso_model = Lasso(alpha=1.0, max_iter=1000)
lasso_model.fit(X, y)

# Hồi quy Ridge
ridge_model = Ridge(alpha=1.0, max_iter=1000)
ridge_model.fit(X, y)

# Neural Network Regression với ReLU
neural_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, activation='relu', solver='adam', random_state=0)
neural_model.fit(X, y)

# Hàm dự đoán cho hồi quy tuyến tính
def predict_gold_price_linear(open_value: float, vol_value: float) -> float:
    return linear_model.predict(np.array([[open_value, vol_value]]))[0]

# Hàm dự đoán cho Lasso
def predict_gold_price_lasso(open_value: float, vol_value: float) -> float:
    return lasso_model.predict(np.array([[open_value, vol_value]]))[0]

# Hàm dự đoán cho Ridge
def predict_gold_price_ridge(open_value: float, vol_value: float) -> float:
    return ridge_model.predict(np.array([[open_value, vol_value]]))[0]

# Hàm dự đoán cho Neural Network
def predict_gold_price_neural(open_value: float, vol_value: float) -> float:
    return neural_model.predict(np.array([[open_value, vol_value]]))[0]

# Định nghĩa lớp dữ liệu đầu vào
class PredictionInput(BaseModel):
    open: float
    vol: float

# Endpoint giá vàng
@app.post("/predict")
async def predict(input_data: PredictionInput):
    open_value = input_data.open
    vol_value = input_data.vol

    try:
        # Dự đoán giá vàng
        predicted_linear = predict_gold_price_linear(open_value, vol_value)
        predicted_lasso = predict_gold_price_lasso(open_value, vol_value)
        predicted_ridge = predict_gold_price_ridge(open_value, vol_value)
        predicted_neural = predict_gold_price_neural(open_value, vol_value)

        return {
            "predicted_linear": predicted_linear,
            "predicted_lasso": predicted_lasso,
            "predicted_ridge": predicted_ridge,
            "predicted_neural": predicted_neural
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
    </head>
    <body>
        <div class="container">
            <h2 class="text-center">Dự đoán giá vàng</h2>
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
            <div id="result" style="margin-top: 20px;"></div>
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
                                <li>Hồi quy Tuyến tính: ${data.predicted_linear.toFixed(2)} VNĐ</li>
                                <li>Hồi quy Lasso: ${data.predicted_lasso.toFixed(2)} VNĐ</li>
                                <li>Hồi quy Ridge: ${data.predicted_ridge.toFixed(2)} VNĐ</li>
                                <li>Mạng nơ-ron: ${data.predicted_neural.toFixed(2)} VNĐ</li>
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

# Chạy ứng dụng
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
