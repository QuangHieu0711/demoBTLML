import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fastapi import HTTPException
from sklearn.linear_model import Lasso
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import uvicorn

app = FastAPI()

df = pd.read_csv("Gia_Vang_2018_2020.csv")
df.columns = ["Date", "Price", "Open", "Vol"]

# Tiền xử lý dữ liệu
def preprocess_data(df):
    df.dropna(inplace=True)
    df.drop(columns=['Date', 'Vol'], inplace=True)
    return df

df = preprocess_data(df)
X = df[['Open']].values
y = df['Price'].values
alpha = 0.1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def linear_regression(X, y):
    x = X[:, 0]
    N = len(y)
    m = (N * np.sum(x * y) - np.sum(x) * np.sum(y)) / (N * np.sum(x ** 2) - (np.sum(x) ** 2))
    b = np.mean(y) - (m * np.mean(x))
    return m, b

m, b = linear_regression(X, y)

def predict_gold_price_linear(open_value: float) -> float:
    return m * open_value + b

N = len(y)
x = X[:, 0]
lasso_m = (N * np.sum(x * y) - np.sum(x) * np.sum(y)) / (N * np.sum(x ** 2) + alpha - (np.sum(x) ** 2))
lasso_b = np.mean(y) - (lasso_m * np.mean(x))

def predict_gold_price_lasso(open_value: float) -> float:
    return lasso_m * open_value + lasso_b

lasso_model = Lasso(alpha=alpha, max_iter=1000)
lasso_model.fit(X, y)

def predict_gold_price_lasso_sklearn(open_value: float) -> float:
    return lasso_model.predict(np.array([[open_value]]))[0]

neural_model = MLPRegressor(hidden_layer_sizes=(500, 300), max_iter=1000, activation='relu', solver='adam', random_state=0)
neural_model.fit(X, y)

def predict_gold_price_neural(open_value: float) -> float:
    return neural_model.predict(np.array([[open_value]]))[0]

def bagging_regression(X, y):
    base_model = MLPRegressor(hidden_layer_sizes=(500,300), max_iter=1000, random_state=42)
    bagging_model = BaggingRegressor(estimator=base_model, n_estimators=10, random_state=42)
    bagging_model.fit(X, y)
    return bagging_model

bagging_model = bagging_regression(X, y)

def predict_gold_price_bagging(open_value: float) -> float:
    return bagging_model.predict([[open_value]])[0]

def predict_gold_price_combined(open_value: float) -> float:
    pred_linear = predict_gold_price_linear(open_value)
    pred_lasso = predict_gold_price_lasso(open_value)
    pred_neural = predict_gold_price_neural(open_value)
    return np.mean([pred_linear, pred_lasso, pred_neural])

class PredictionInput(BaseModel):
    open: float

@app.post("/predict")
async def predict(input_data: PredictionInput):
    open_value = input_data.open

    try:
        predicted_linear = predict_gold_price_linear(open_value)
        predicted_lasso = predict_gold_price_lasso(open_value)
        predicted_lasso_sklearn = predict_gold_price_lasso_sklearn(open_value)
        predicted_neural = predict_gold_price_neural(open_value)
        predicted_bagging = predict_gold_price_bagging(open_value)
        predicted_combined = predict_gold_price_combined(open_value)

        mse_linear = mean_squared_error(y, [predict_gold_price_linear(X[i, 0]) for i in range(len(y))])
        r2_linear = r2_score(y, [predict_gold_price_linear(X[i, 0]) for i in range(len(y))])

        mse_lasso = mean_squared_error(y, [predict_gold_price_lasso(X[i, 0]) for i in range(len(y))])
        r2_lasso = r2_score(y, [predict_gold_price_lasso(X[i, 0]) for i in range(len(y))])

        mse_lasso_tv = mean_squared_error(y, lasso_model.predict(X))
        r2_lasso_tv = r2_score(y, lasso_model.predict(X))

        mse_nn = mean_squared_error(y, neural_model.predict(X))
        r2_nn = r2_score(y, neural_model.predict(X))

        mse_bagging = mean_squared_error(y, bagging_model.predict(X))
        r2_bagging = r2_score(y, bagging_model.predict(X))

        mse_bagging_kethop = mean_squared_error(y, [predict_gold_price_combined(X[i, 0]) for i in range(len(y))])
        r2_bagging_kethop = r2_score(y, [predict_gold_price_combined(X[i, 0]) for i in range(len(y))])

        return {
            "Kết quả dự đoán": {
                "Giá vàng dự đoán theo Hồi quy tuyến tính": predicted_linear,
                "Giá vàng dự đoán theo Hồi quy Lasso": predicted_lasso,
                "Giá vàng dự đoán theo Hồi quy Lasso (sklearn)": predicted_lasso_sklearn,
                "Giá vàng dự đoán theo Neural Network (ReLU)": predicted_neural,
                "Giá vàng dự đoán theo Bagging": predicted_bagging,
                "Giá vàng dự đoán theo cách kết hợp": predicted_combined
            },
            "MSE và R²": {
                "MSE Hồi quy tuyến tính": mse_linear,
                "R² Hồi quy tuyến tính": r2_linear,
                "MSE Hồi quy Lasso": mse_lasso,
                "R² Hồi quy Lasso": r2_lasso,
                "MSE Hồi quy Lasso (sklearn)": mse_lasso_tv,
                "R² Hồi quy Lasso(sklearn)": r2_lasso_tv,
                "MSE Neural Network (ReLU)": mse_nn,
                "R² Neural Network (ReLU)": r2_nn,
                "MSE Bagging": mse_bagging,
                "R² Bagging": r2_bagging,
                "MSE Bagging (Kết hợp)": mse_bagging_kethop,
                "R² Bagging (Kết hợp)": r2_bagging_kethop
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
