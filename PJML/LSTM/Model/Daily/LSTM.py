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
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras import regularizers # type: ignore
from tensorflow.keras.losses import Huber # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau# type: ignore
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error ,r2_score


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
    weather_columns = ['weather_Mostly_Sunny', 'weather_Partly_Cloudy', 'weather_Scattered_Shower']
    event_column = ['event_Normal_Day']

    df[weather_columns] = df[weather_columns].astype(int)
    df[event_column] = df[event_column].astype(int)

    # สเกลข้อมูลให้อยู่ในช่วง [0, 1] สำหรับคอลัมน์ sales_amount
    scaler = MinMaxScaler()
    df['sales_amount_scaled'] = scaler.fit_transform(df[['sales_amount']])

    # เติมค่าขาดหายไปในคอลัมน์ Temperature (ถ้ามี)
    if 'Temperature' in df.columns:
        df['Temperature'] = df['Temperature'].fillna(df['Temperature'].mean())

    # ✅ Data Augmentation (เพิ่ม Noise เล็กน้อยเพื่อช่วยลด Overfitting)
    df['sales_amount_scaled'] += np.random.normal(0, 0.01, df.shape[0])

    # เตรียมข้อมูลสำหรับ LSTM
    sequence_length = 60
    X, y = [], []
    features = ['Temperature', 'Day', 'Month', 'Year'] + weather_columns + event_column
    for i in range(sequence_length, len(df)):
        X.append(df.iloc[i-sequence_length:i][features].values)
        y.append(df['sales_amount_scaled'].iloc[i])

    return np.array(X), np.array(y), scaler, df

#===============================================เทรน=========================================
def train_lstm_model1(X_train, y_train, X_val, y_val):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'ModelLstm1')

    model_path1 = os.path.join(model_dir, 'lstm_model1.pkl')
    date_path = os.path.join(model_dir, 'last_trained_date1.pkl')

    os.makedirs(model_dir, exist_ok=True)

    if os.path.exists(model_path1):
        model = joblib.load(model_path1)
        print("📥 โหลดโมเดลจากไฟล์ที่เก็บไว้แล้ว")

        if os.path.exists(date_path):
            last_trained_date = joblib.load(date_path)
        else:
            last_trained_date = datetime.min

        if datetime.now() - last_trained_date < timedelta(days=30):
            print("⏳ ยังไม่ถึงเวลาเทรนใหม่ (ต้องรออย่างน้อย 30 วัน)")
            return model
        else:
            print("🔄 ถึงเวลาเทรนโมเดลใหม่!")

    # สร้าง Callback Logger เพื่อแจ้งเตือนเป็นภาษาไทย
    class EarlyStoppingLogger(Callback):
        def on_train_end(self, logs=None):
            print("⚠️ การฝึกโมเดลหยุดก่อนครบกำหนด เนื่องจากโมเดลอาจเกิดการ Overfitting!")

    early_stopping = EarlyStopping(
        monitor='val_loss',  # ✅ เปลี่ยนจาก loss -> val_loss
        patience=5,
        restore_best_weights=True
    )

    # ✅ Learning Rate Schedule
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1
    )

    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.3),
        LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)),  # ✅ เพิ่ม Regularization
        Dropout(0.3),
        LSTM(64, kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),  # ✅ Regularization
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.0005), loss=Huber(), metrics=['mae', 'mape'])
    print("🛠️ กำลังสร้างโมเดลใหม่...")

    model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val),  # ✅ ใช้ Validation
                        epochs=30, batch_size=32, verbose=1, 
                        callbacks=[early_stopping, reduce_lr, EarlyStoppingLogger()])

    joblib.dump(model, model_path1)
    joblib.dump(datetime.now(), date_path)
    print("✅ บันทึกโมเดลและวันที่เทรนล่าสุดเรียบร้อยแล้ว")

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
    df = load_data()
    df_preprocessed = preprocess_data(df)

    X, y, scaler, df_prepared = prepare_data(df_preprocessed)

<<<<<<< HEAD
    # ✅ แบ่งข้อมูลเป็น Train, Validation และ Test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
=======
    # แบ่งข้อมูลเป็น Train และ Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
>>>>>>> 5a8c55dd4335f0ff13cf6d49a6dc7523735097e7

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

    model = train_lstm_model1(X_train, y_train, X_val, y_val)

    predicted_sales = model.predict(X_test)
    predicted_sales = scaler.inverse_transform(predicted_sales)

    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(predicted_sales, y_test_original)
    mape = mean_absolute_percentage_error(predicted_sales, y_test_original)
    r2 = r2_score(y_test_original, predicted_sales)

    predicted_date = df_prepared['sale_date'].iloc[-1] + pd.DateOffset(days=1)

    return jsonify({
        'predicted_sales': float(predicted_sales[-1][0]),
        'predicted_date': str(predicted_date),
        'model_name': "LSTM",
        'mae': mae,
        'mape': mape,
        'r2': r2
    })

if __name__ == '__main__':
    app.run(host='localhost', port=8885, debug=True)