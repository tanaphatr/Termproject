import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from flask import Flask, jsonify
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras import regularizers  # type: ignore
from tensorflow.keras.losses import Huber  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# เพิ่ม path ของโปรเจค
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'PJML')))

from Datafile.load_data import load_dataps
from Preprocess.preprocess_data import preprocess_dataps

#===============================================Time series=========================================
def create_dataset(data, time_step=12):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step)])
        y.append(data[i + time_step, 0])  # ใช้ Monthly_Total_Quantity
    return np.array(X), np.array(y)

#===============================================Stock=========================================
def train_lstm_model2(X_train, y_train, product_code, input_shape=None):
    # ระบุโฟลเดอร์ปัจจุบัน
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'ModelLstm2')
    
    # สร้าง path ของไฟล์โมเดลและวันที่เทรน
    model_path2 = os.path.join(model_dir, f'lstm_model_{product_code}.pkl')
    date_path = os.path.join(model_dir, f'last_trained_date_{product_code}.pkl')
    
    # ตรวจสอบว่าโฟลเดอร์สำหรับเก็บโมเดลมีอยู่หรือไม่ ถ้าไม่มีให้สร้างใหม่
    os.makedirs(model_dir, exist_ok=True)
    
    # ตรวจสอบว่าไฟล์โมเดลมีอยู่หรือไม่
    if os.path.exists(model_path2):
        # ถ้ามีไฟล์โมเดลให้โหลดไฟล์ที่มีอยู่
        model = joblib.load(model_path2)
        print(f"โหลดโมเดลจากไฟล์ที่เก็บไว้สำหรับ Product_code: {product_code}")
        
        # โหลดวันที่เทรนล่าสุด
        if os.path.exists(date_path):
            last_trained_date = joblib.load(date_path)
        else:
            last_trained_date = datetime.min  # กำหนดค่าเริ่มต้นหากไม่มีไฟล์วันที่
        
        # เช็คว่าเวลาผ่านไป 30 วันหรือยัง
        if datetime.now() - last_trained_date < timedelta(days=30):
            print("โมเดลยังไม่ถึงเวลาในการเทรนใหม่")
            return model
        else:
            print("ถึงเวลาในการเทรนใหม่")
    else:
        # ถ้าไม่มีไฟล์โมเดล จะสร้างโมเดลใหม่
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        print(f"สร้างโมเดลใหม่สำหรับ Product_code: {product_code}")
    
    # เทรนโมเดล
    model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=1)
    
    # บันทึกโมเดลและวันที่เทรนล่าสุด
    joblib.dump(model, model_path2)
    joblib.dump(datetime.now(), date_path)
    print(f"บันทึกโมเดลและวันที่เทรนล่าสุดสำหรับ Product_code: {product_code}")
    
    return model

#===============================================predict_next_month=========================================
def predict_next_month(model, scaled_data, scaler):
    latest_data = scaled_data[-12:]  # ใช้ข้อมูล 12 เดือนสุดท้าย
    latest_data = latest_data.reshape((1, latest_data.shape[0], 1))  # ปรับรูปแบบข้อมูล
    next_month_prediction = model.predict(latest_data)

    # เติมค่าให้เหมาะสมกับการ inverse transform
    next_month_full = np.zeros((1, scaled_data.shape[1]))
    next_month_full[:, 0] = next_month_prediction[:, 0]
    return scaler.inverse_transform(next_month_full)

# Flask App
app = Flask(__name__)

@app.route('/', methods=['GET'])
def predict_sales():
    # เตรียมข้อมูลเพิ่มเติม
    dfps = load_dataps()
    dfps = preprocess_dataps(dfps)
    dfps = dfps[['Product_code', 'Year', 'Month', 'Monthly_Total_Quantity']]

    # สร้างเดือนถัดไป
    last_row = dfps.iloc[-1]
    last_month, last_year = int(last_row['Month']), int(last_row['Year'])
    next_month = (last_month % 12) + 1
    next_year = last_year if next_month != 1 else last_year + 1


    scaler = MinMaxScaler(feature_range=(0, 1))
    predictions = []

    grouped = dfps.groupby('Product_code')
    for product_code, product_data in grouped:
        try:
            # เรียงลำดับข้อมูลตามปีและเดือน
            product_data = product_data.sort_values(by=['Year', 'Month'])
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

            predictions.append({
                "Product_code": product_code,
                "Prediction": next_month_prediction[0, 0]
            })

        except Exception as e:
            print(f"Error processing Product_code {product_code}: {e}")
            continue

    # รวมข้อมูลเพื่อส่งกลับ
    response_data = {
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