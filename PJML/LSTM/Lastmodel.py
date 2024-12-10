import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, jsonify
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense  # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# เพิ่ม path ของโปรเจค
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'PJML')))

from Datafile.load_data import load_dataps
from Preprocess.preprocess_data import preprocess_dataps
from Datafile.load_data import load_data

# เตรียมข้อมูล
def prepare_data(df):
    df['sales_amount'] = df['sales_amount'].fillna(df['sales_amount'].mean())
    if 'sale_date' in df.columns:
        df['sale_date'] = df['sale_date'].astype(str)
        df['sale_date'] = df['sale_date'].apply(lambda x: str(int(x[:4]) - 543) + x[4:] if len(x) == 10 else x)
        df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
        df = df.sort_values('sale_date')

    scaler = MinMaxScaler()
    df['sales_amount_scaled'] = scaler.fit_transform(df[['sales_amount']])

    sequence_length = 20
    X, y = [], []
    for i in range(sequence_length, len(df)):
        X.append(df['sales_amount_scaled'].iloc[i-sequence_length:i].values)
        y.append(df['sales_amount_scaled'].iloc[i])

    df[['sale_date']].to_csv('sale_dates.csv', index=False)
    return np.array(X), np.array(y), scaler, df

# Train LSTM Model
def train_lstm_model1(X, y):
    model_path1 = r'.\ModelLstm\lstm_model1.pkl'
    if os.path.exists(model_path1):
        model = joblib.load(model_path1)
        last_trained_date = joblib.load(r'.\ModelLstm\last_trained_date1.pkl')
        if datetime.now() - last_trained_date < timedelta(days=30):
            return model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=32, verbose=1)
    joblib.dump(model, model_path1)
    joblib.dump(datetime.now(), r'.\ModelLstm\last_trained_date1.pkl')
    return model

# Predict Next Sales
def predict_next_sales(model, X, scaler, df):
    last_sequence = X[-1].reshape(1, -1, 1)
    prediction_scaled = model.predict(last_sequence)
    predicted_sales = scaler.inverse_transform(prediction_scaled)[0][0]
    predicted_date = df['sale_date'].iloc[-1] + pd.DateOffset(days=1)
    return predicted_sales, predicted_date

# Time Series Data
def create_dataset(data, time_step=30):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i+time_step)])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Train Stock Prediction Model
def train_lstm_model2(X_train, y_train, product_code, input_shape=None):
    model_path2 = f'.\ModelLstm\lstm_model_{product_code}.pkl'
    if os.path.exists(model_path2):
        model = joblib.load(model_path2)
        last_trained_date = joblib.load(f'.\ModelLstm\last_trained_date_{product_code}.pkl')
        if datetime.now() - last_trained_date < timedelta(days=30):
            return model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
    joblib.dump(model, model_path2)
    joblib.dump(datetime.now(), f'.\ModelLstm\last_trained_date_{product_code}.pkl')
    return model

# Predict Next Month Sales
def predict_next_month(model, scaled_data, scaler):
    latest_data = scaled_data[-30:]
    latest_data = latest_data.reshape((1, latest_data.shape[0], 1))
    next_month_prediction = model.predict(latest_data)
    next_month_full = np.zeros((1, scaled_data.shape[1]))
    next_month_full[:, 0] = next_month_prediction[:, 0]
    return scaler.inverse_transform(next_month_full)

# API for Prediction
app = Flask(__name__)

@app.route('/', methods=['GET'])
def predict_sales():
    df = load_data()

    X, y, scaler, df_prepared = prepare_data(df)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # แบ่งข้อมูลเป็น training, validation และ testing set
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

    model = train_lstm_model1(X_train, y_train)

    # ทำนายยอดขายในวันถัดไป
    predicted_sales = model.predict(X_test)
    predicted_sales = scaler.inverse_transform(predicted_sales)

    mse = mean_squared_error(y_test, predicted_sales)
    mae = mean_absolute_error(y_test, predicted_sales)
    mape = mean_absolute_percentage_error(y_test, predicted_sales)

    # เพิ่มข้อมูลเพิ่มเติม
    dfps = load_dataps()
    dfps = preprocess_dataps(dfps)
    dfps = dfps[['Product_code', 'Year', 'Month', 'Monthly_Total_Quantity']]

    scaler = MinMaxScaler(feature_range=(0, 1))
    predictions = []

    grouped = dfps.groupby('Product_code')
    for product_code, product_data in grouped:
        try:
            scaled_product_data = scaler.fit_transform(product_data[['Monthly_Total_Quantity']])
            X, y = create_dataset(scaled_product_data)
            if len(X) == 0:
                continue

            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            model = train_lstm_model2(X_train, y_train, product_code, input_shape=(X_train.shape[1], X_train.shape[2]))

            next_month_prediction = predict_next_month(model, scaled_product_data, scaler)

            last_month = int(dfps['Month'].iloc[-1])
            last_year = int(dfps['Year'].iloc[-1])
            next_month = (last_month % 12) + 1
            next_year = last_year if next_month != 1 else last_year + 1

            predictions.append({
                "Product_code": product_code,
                "Next_Year": next_year,
                "Next_Month": next_month,
                "Prediction": next_month_prediction[0, 0]
            })

        except Exception as e:
            print(f"Error processing Product_code {product_code}: {e}")
            continue

    response_data = {
        'predicted_sales': float(predicted_sales[-1][0]),
        'predicted_date': str(df_prepared['sale_date'].iloc[-1] + pd.DateOffset(days=1)),
        'model_name': "LSTM",
        'mae': mae,
        'mape': mape,
        'predictions': [
            {"Next_Year": next_year, "Next_Month": next_month}
        ] + [
            {"Product_code": prediction["Product_code"], "Prediction": round(prediction["Prediction"], 0)}
            for prediction in predictions
            if prediction["Prediction"] != 0.0 and prediction["Product_code"] != ""
        ]
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='localhost', port=8887, debug=True)