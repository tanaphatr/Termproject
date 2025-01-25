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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# เพิ่ม path ของโปรเจค
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'PJML')))

from Datafile.load_data import load_data
from Preprocess.preprocess_data import preprocess_data

#===============================================เตรียมข้อมูล=========================================
def prepare_data(df):
    # ตรวจสอบและจัดการค่า NaT ใน sale_date
    df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
    df = df.dropna(subset=['sale_date'])

    # แปลงค่าฟีเจอร์ weather และ event ให้อยู่ในรูปของตัวเลข
    weather_columns = ['weather_Mostly Sunny', 'weather_Partly Cloudy', 'weather_Scattered Shower']
    event_column = ['event_Normal Day']

    df[weather_columns] = df[weather_columns].astype(int)
    df[event_column] = df[event_column].astype(int)

    # สเกลข้อมูลให้อยู่ในช่วง [0, 1] สำหรับคอลัมน์ sales_amount
    scaler = MinMaxScaler()
    df['sales_amount_scaled'] = scaler.fit_transform(df[['sales_amount']])

    # เติมค่าขาดหายไปในคอลัมน์ Temperature (ถ้ามี)
    if 'Temperature' in df.columns:
        df['Temperature'] = df['Temperature'].fillna(df['Temperature'].mean())

    # เตรียมข้อมูลสำหรับ LSTM
    sequence_length = 60
    X, y = [], []
    for i in range(sequence_length, len(df)):
        features = ['Temperature', 'Date', 'Month', 'Year' ] + weather_columns + event_column
        X.append(df.iloc[i-sequence_length:i][features].values)
        y.append(df['sales_amount_scaled'].iloc[i])

    return np.array(X), np.array(y), scaler, df

#===============================================เทรน=========================================
def train_lstm_model1(X_train, y_train):
    # ระบุ path ของโฟลเดอร์ปัจจุบัน
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'ModelLstm1')
    
    # สร้าง path ของไฟล์โมเดลและวันที่เทรน
    model_path1 = os.path.join(model_dir, 'lstm_model1.pkl')
    date_path = os.path.join(model_dir, 'last_trained_date1.pkl')
    
    # ตรวจสอบว่าโฟลเดอร์สำหรับเก็บโมเดลมีอยู่หรือไม่ ถ้าไม่มีให้สร้างใหม่
    os.makedirs(model_dir, exist_ok=True)
    
    # ตรวจสอบว่าไฟล์โมเดลมีอยู่หรือไม่
    if os.path.exists(model_path1):
        # ถ้ามีไฟล์โมเดลให้โหลดไฟล์ที่มีอยู่
        model = joblib.load(model_path1)
        print("โหลดโมเดลจากไฟล์ที่เก็บไว้")
        
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
            LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), kernel_regularizer=regularizers.l2(0.001)),
            Dropout(0.3),  # อาจปรับเป็น 0.5 หรือ 0.2 ดูผล
            LSTM(128),
            Dense(1)
        ])

    model.compile(optimizer=Adam(learning_rate=0.0005), loss=Huber(), metrics=['mae', 'mape'])
    print("สร้างโมเดลใหม่")

    # เทรนโมเดล
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)

    # บันทึกโมเดลและวันที่เทรนล่าสุด
    joblib.dump(model, model_path1)
    joblib.dump(datetime.now(), date_path)
    print("บันทึกโมเดลและวันที่เทรนล่าสุด")
    
    return model

#===============================================ทำนาย=========================================
def predict_next_sales(model, X, scaler, df):
    # ใช้ข้อมูลล่าสุดในการทำนาย
    last_sequence = X[-1].reshape(1, -1, X.shape[2])
    prediction_scaled = model.predict(last_sequence)
    predicted_sales = scaler.inverse_transform(prediction_scaled)[0][0]
    
    # ดึงวันที่ล่าสุดจากคอลัมน์ 'sale_date' และเพิ่มวัน
    predicted_date = df['sale_date'].iloc[-1] + pd.DateOffset(days=1)
    return predicted_sales, predicted_date

#===============================================API=========================================

app = Flask(__name__)

@app.route('/', methods=['GET'])
def predict_sales_api():
    df = load_data()  # เรียกใช้ข้อมูลจาก load_data
    
    # ส่ง df ไปยังฟังก์ชัน preprocess_data
    df_preprocessed = preprocess_data(df)  # ตอนนี้ข้อมูล df จะถูก preprocess แล้ว

    # เตรียมข้อมูลให้เหมาะสมสำหรับ LSTM
    X, y, scaler, df_prepared = prepare_data(df_preprocessed)

    # แบ่งข้อมูลเป็น Train และ Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    # ปรับรูปร่างข้อมูลให้เหมาะสมกับ LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

    # ฝึกโมเดล
    model = train_lstm_model1(X_train, y_train)
    
    # ทำนายยอดขายในวันถัดไปโดยใช้ข้อมูล testing
    predicted_sales = model.predict(X_test)
    predicted_sales = scaler.inverse_transform(predicted_sales)

    # แปลง y_test กลับจากการสเกล
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

    # คำนวณ mse สำหรับ Testing Data
    mae = mean_absolute_error(predicted_sales,y_test_original)
    mape = mean_absolute_percentage_error(predicted_sales,y_test_original)

    # ตรวจสอบวันที่ล่าสุด
    predicted_date = df_prepared['sale_date'].iloc[-1] + pd.DateOffset(days=1)

    return jsonify({
        'predicted_sales': float(predicted_sales[-1][0]),
        'predicted_date': str(predicted_date),
        'model_name': "LSTM",
        'mae': mae,
        'mape': mape,
    })

if __name__ == '__main__':
    app.run(host='localhost', port=8885, debug=True)
