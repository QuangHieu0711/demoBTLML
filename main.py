from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fastapi import HTTPException
from sklearn.linear_model import Lasso
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

app = FastAPI()

# Đọc dữ liệu từ tệp CSV
df = pd.read_csv("Gia_Vang_2019_2022.csv")
df.columns = ["Date", "Price", "Open", "Vol"]

# Tiền xử lý dữ liệu
def preprocess_data(df):
    df.dropna(inplace=True)  # Loại bỏ các hàng chứa giá trị null
    df.drop(columns=['Date', 'Vol'], inplace=True)  # Xóa cột 'Date' và 'Vol'
    return df

# Tiền xử lý dữ liệu
df = preprocess_data(df)

# Chuyển đổi dữ liệu thành mảng NumPy
X = df[['Open']].values  # Chỉ giữ lại cột 'Open'
y = df['Price'].values
alpha = 0.1  # Tham số alpha cho hồi quy Lasso

# Tính toán tham số hồi quy tuyến tính
def linear_regression(X, y):
    x = X[:, 0]  # Giá mở cửa

    N = len(y)
    m = (N * np.sum(x * y) - np.sum(x) * np.sum(y)) / (N * np.sum(x ** 2) - (np.sum(x) ** 2))
    b = np.mean(y) - (m * np.mean(x))
    return m, b

# Lấy tham số hồi quy tuyến tính
m, b = linear_regression(X, y)

# Dự đoán giá vàng bằng hồi quy tuyến tính
def predict_gold_price_linear(open_value: float) -> float:
    return m * open_value + b

# Hồi quy Lasso (tính toán thủ công)
N = len(y)
x = X[:, 0]  # Giá mở cửa

# Tính toán tham số hồi quy Lasso
lasso_m = (N * np.sum(x * y) - np.sum(x) * np.sum(y)) / (N * np.sum(x ** 2) + alpha - (np.sum(x) ** 2))
lasso_b = np.mean(y) - (lasso_m * np.mean(x))

# Hàm dự đoán cho hồi quy Lasso
def predict_gold_price_lasso(open_value: float) -> float:
    return lasso_m * open_value + lasso_b

# Sử dụng hàm Lasso trong thư viện scikit-learn
lasso_model = Lasso(alpha=alpha, max_iter=1000)
lasso_model.fit(X, y)

# Hàm dự đoán cho Lasso (scikit-learn)
def predict_gold_price_lasso_sklearn(open_value: float) -> float:
    return lasso_model.predict(np.array([[open_value]]))[0]

# Mạng nơ-ron hồi quy với ReLU
neural_model = MLPRegressor(hidden_layer_sizes=(500, 300), max_iter=1000, activation='relu', solver='adam', random_state=0)
neural_model.fit(X, y)

# Hàm dự đoán cho mạng nơ-ron
def predict_gold_price_neural(open_value: float) -> float:
    return neural_model.predict(np.array([[open_value]]))[0]

# Hàm bagging
def bagging_regression(X, y):
    base_model = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
    bagging_model = BaggingRegressor(estimator=base_model, n_estimators=10, random_state=42)
    bagging_model.fit(X, y)
    return bagging_model

# Tính toán tham số bagging
bagging_model = bagging_regression(X, y)

# Dự đoán giá vàng bằng bagging
def predict_gold_price_bagging(open_value: float) -> float:
    return bagging_model.predict([[open_value]])[0]

class PredictionInput(BaseModel):
    open: float

# Endpoint dự đoán giá vàng
@app.post("/predict")
async def predict(input_data: PredictionInput):
    open_value = input_data.open

    try:
        # Dự đoán giá vàng
        predicted_linear = predict_gold_price_linear(open_value)
        predicted_lasso = predict_gold_price_lasso(open_value)
        predicted_lasso_sklearn = predict_gold_price_lasso_sklearn(open_value)
        predicted_neural = predict_gold_price_neural(open_value)
        predicted_bagging = predict_gold_price_bagging(open_value)

        # Tính MSE và R^2 cho từng phương pháp
        mse_linear = mean_squared_error(y, [predict_gold_price_linear(X[i, 0]) for i in range(len(y))])
        r2_linear = r2_score(y, [predict_gold_price_linear(X[i, 0]) for i in range(len(y))])

        mse_lasso = mean_squared_error(y, lasso_model.predict(X))
        r2_lasso = r2_score(y, lasso_model.predict(X))

        mse_nn = mean_squared_error(y, neural_model.predict(X))
        r2_nn = r2_score(y, neural_model.predict(X))

        mse_bagging = mean_squared_error(y, bagging_model.predict(X))
        r2_bagging = r2_score(y, bagging_model.predict(X))

        # Trả về kết quả theo định dạng mong muốn
        return {
            "Kết quả dự đoán": {
                "Giá vàng dự đoán theo Hồi quy tuyến tính": predicted_linear,
                "Giá vàng dự đoán theo Hồi quy Lasso": predicted_lasso,
                "Giá vàng dự đoán theo Hồi quy Lasso (sklearn)": predicted_lasso_sklearn,
                "Giá vàng dự đoán theo Neural Network (ReLU)": predicted_neural,
                "Giá vàng dự đoán theo Bagging": predicted_bagging
            },
            "MSE và R²": {
                "MSE Hồi quy tuyến tính": mse_linear,
                "R² Hồi quy tuyến tính": r2_linear,
                "MSE Hồi quy Lasso": mse_lasso,
                "R² Hồi quy Lasso": r2_lasso,
                "MSE Neural Network (ReLU)": mse_nn,
                "R² Neural Network (ReLU)": r2_nn,
                "MSE Bagging": mse_bagging,
                "R² Bagging": r2_bagging
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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
                background-color: #f8f8f8;
            }
            .container {
                display: flex;
                justify-content: space-between;
                width: 80%;
            }
            .form-container {
                background-color: #f0f0f0;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                width: 48%;
            }
            .result-container {
                background-color: #f0f0f0;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                width: 48%;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="form-container">
                <h2 class="text-center">Dự đoán giá vàng</h2>
                <form id="predictionForm">
                    <div class="form-group">
                        <label for="open">Giá mở cửa:</label>
                        <input type="number" id="open" class="form-control" step="any" required>
                    </div>
                    <button type="button" class="btn btn-primary" onclick="predict()">Dự đoán</button>
                </form>
            </div>
            <div class="result-container" id="result">
                <h4>Kết quả dự đoán:</h4>
            </div>
        </div>

        <script>
            async function predict() {
                const open = parseFloat(document.getElementById("open").value);
                const response = await fetch("/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ open: open })
                });
                const data = await response.json();
                const resultDiv = document.getElementById("result");
                resultDiv.innerHTML = "<h4>Kết quả dự đoán:</h4>";
                for (const [key, value] of Object.entries(data["Kết quả dự đoán"])) {
                    resultDiv.innerHTML += `<p>${key}: ${value.toFixed(4)}</p>`;
                }
                resultDiv.innerHTML += "<h4>MSE và R²:</h4>";
                for (const [key, value] of Object.entries(data["MSE và R²"])) {
                    resultDiv.innerHTML += `<p>${key}: ${value.toFixed(4)}</p>`;
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
