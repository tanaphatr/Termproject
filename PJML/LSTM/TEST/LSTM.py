import os
import sys
import pandas as pd
import numpy as np
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'PJML')))

from Datafile.load_data import load_data
from Preprocess.preprocess_data import preprocess_data

#===============================================เตรียมข้อมูล=========================================
def prepare_data(df):
    # ตรวจสอบและจัดการค่า NaT ใน sale_date
    df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
    df = df.dropna(subset=['sale_date'])

    # แยก sale_date เป็น Day, Month, Year
    df['Day'] = df['sale_date'].dt.day
    df['Month'] = df['sale_date'].dt.month
    df['Year'] = df['sale_date'].dt.year

    # แปลงค่าฟีเจอร์ weather และ event ให้อยู่ในรูปของตัวเลข
    weather_columns = ['weather_Mostly Sunny', 'weather_Partly Cloudy', 'weather_Scattered Shower']
    event_column = ['event_Normal Day']

    df[weather_columns] = df[weather_columns].astype(int)
    df[event_column] = df[event_column].astype(int)

    # กรอง outliers
    Q1 = df['sales_amount'].quantile(0.25)
    Q3 = df['sales_amount'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['sales_amount'] >= (Q1 - 1.5 * IQR)) & (df['sales_amount'] <= (Q3 + 1.5 * IQR))]

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
        features = ['Temperature', 'Day', 'Month', 'Year'] + weather_columns + event_column
        X.append(df.iloc[i-sequence_length:i][features].values)
        y.append(df['sales_amount_scaled'].iloc[i])

    return np.array(X), np.array(y), scaler, df

#===============================================เทรน=========================================
def train_lstm_model(X_train, y_train):
    # สร้างโมเดล LSTM
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), kernel_regularizer=regularizers.l2(0.01)),
        Dropout(0.3),  # ใช้อัตรา Dropout ที่ต่ำลง
        LSTM(128),  # ลดขนาดของ LSTM layer
        Dense(1)
    ])

    # ใช้ learning rate ที่ต่ำลง
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=Huber(), metrics=['mae', 'mape'])
    print("สร้างโมเดลใหม่")

    # ใช้ EarlyStopping เพื่อหยุดการฝึกเมื่อโมเดลไม่พัฒนา
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # เทรนโมเดล
    model.fit(X_train, y_train, epochs=3, batch_size=32, validation_split=0.3, callbacks=[early_stopping])
    return model

#===============================================ทำนาย=========================================
def predict_sales(model, X, scaler, df):
    last_sequence = X[-1].reshape(1, -1, X.shape[2])
    prediction_scaled = model.predict(last_sequence)
    predicted_sales = scaler.inverse_transform(prediction_scaled)[0][0]
    
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # ปรับรูปร่างข้อมูลให้เหมาะสมกับ LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

    # เทรนโมเดล
    model = train_lstm_model(X_train, y_train)

    # ทำนายค่า sales สำหรับ test data
    predicted_sales_all = model.predict(X_test)
    predicted_sales_all = scaler.inverse_transform(predicted_sales_all)
    
    # แปลง y_test กลับจากการสเกล
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

    # คำนวณ MAE และ MAPE
    mae = mean_absolute_error(y_test_original, predicted_sales_all)
    mape = mean_absolute_percentage_error(y_test_original, predicted_sales_all)

    # ทำนายวันถัดไป
    predicted_sales, predicted_date = predict_sales(model, X, scaler, df_prepared)

    return jsonify({
        'predicted_sales': float(predicted_sales),
        'predicted_date': predicted_date,
        'model_name': "LSTM",
        'mae': mae,
        'mape': mape,
    })

if __name__ == '__main__':
    app.run(host='localhost', port=8885, debug=True)
