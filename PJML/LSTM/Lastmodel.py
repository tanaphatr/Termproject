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
from tensorflow.keras.layers import LSTM, Dense ,Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras import regularizers # type: ignore
from tensorflow.keras.losses import Huber # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error, mean_absolute_percentage_error

# เพิ่ม path ของโปรเจค
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'PJML')))

from Datafile.load_data import load_dataps
from Preprocess.preprocess_data import preprocess_dataps
from Datafile.load_data import load_data

# prepare_data ทำความสะอาดข้อมูล, เติมค่าขาดหาย, แปลงวันที่, และสเกลข้อมูลสำหรับการฝึกโมเดล
# train_lstm_model1 เทรนโมเดล LSTM สำหรับการทำนายยอดขาย
# predict_next_sales ทำนายยอดขายในวันถัดไปโดยใช้โมเดล LSTM
# create_dataset สร้างชุดข้อมูลสำหรับฝึกโมเดล โดยการแยกข้อมูลเป็นลำดับเวลา
# train_lstm_model2 เทรนโมเดล LSTM สำหรับการทำนายปริมาณสินค้าตามลำดับเวลา
# predict_next_month ทำนายยอดขายสำหรับเดือนถัดไปโดยใช้โมเดล LSTM
# predict_sales API endpoint สำหรับทำนายยอดขายจากข้อมูลที่มี โดยใช้ LSTM
#===============================================เตรียมข้อมูล=========================================
def prepare_data(df):
    # เติมค่าขาดหายไปในคอลัมน์เป้าหมาย
    df['sales_amount'] = df['sales_amount'].fillna(df['sales_amount'].median())

    # แปลงวันที่จาก พ.ศ. เป็น ค.ศ.
    if 'sale_date' in df.columns:
        df['sale_date'] = df['sale_date'].astype(str)  # แปลงเป็น string
        df['sale_date'] = df['sale_date'].apply(lambda x: str(int(x[:4]) - 543) + x[4:] if len(x) == 10 else x)

        # แปลงวันที่ให้เป็น datetime และเรียงลำดับ
        df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
        df = df.sort_values('sale_date')

    # สเกลข้อมูลให้อยู่ในช่วง [0, 1]
    scaler = StandardScaler()
    df['sales_amount_scaled'] = scaler.fit_transform(df[['sales_amount']])

    # เตรียมข้อมูลสำหรับ LSTM
    sequence_length = 20  # จำนวนวันใน sequence
    X, y = [], []
    for i in range(sequence_length, len(df)):
        X.append(df['sales_amount_scaled'].iloc[i-sequence_length:i].values)
        y.append(df['sales_amount_scaled'].iloc[i])

    # บันทึกคอลัมน์ sale_date เป็นไฟล์ CSV
    df[['sale_date']].to_csv('sale_dates.csv', index=False)

    return np.array(X), np.array(y), scaler, df
#===============================================Dalisale=========================================
def train_lstm_model1(X, y):
    # ระบุ path ของโฟลเดอร์ปัจจุบัน
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'ModelLstm')
    
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
        LSTM(256, return_sequences=True, input_shape=(X.shape[1], 1), kernel_regularizer=regularizers.l2(0.01)),
        Dropout(0.4),  # เพิ่มค่า Dropout เพื่อป้องกัน overfitting
        LSTM(256),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=Huber(), metrics=['mae', 'mape'])
    print("สร้างโมเดลใหม่")

    # เทรนโมเดล
    model.fit(X, y, epochs=150, batch_size=32, verbose=1)
    
    # บันทึกโมเดลและวันที่เทรนล่าสุด
    joblib.dump(model, model_path1)
    joblib.dump(datetime.now(), date_path)
    print("บันทึกโมเดลและวันที่เทรนล่าสุด")
    
    return model
#===============================================ทำนาย=========================================
def predict_next_sales(model, X, scaler, df):
    # ใช้ข้อมูลล่าสุดในการทำนาย
    last_sequence = X[-1].reshape(1, -1, 1)
    prediction_scaled = model.predict(last_sequence)
    predicted_sales = scaler.inverse_transform(prediction_scaled)[0][0]
    
    # ดึงวันที่ล่าสุดจากคอลัมน์ 'sale_date' และเพิ่มวัน
    predicted_date = df['sale_date'].iloc[-1] + pd.DateOffset(days=1)
    return predicted_sales, predicted_date

#===============================================Time series=========================================
def create_dataset(data, time_step=30):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i+time_step)])
        y.append(data[i + time_step, 0])  # ปริมาณสินค้าที่ต้องเตรียม
    return np.array(X), np.array(y)

#===============================================Stock=========================================
def train_lstm_model2(X_train, y_train, product_code, input_shape=None):
    # ระบุโฟลเดอร์ปัจจุบัน
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'ModelLstm')
    
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
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
    
    # บันทึกโมเดลและวันที่เทรนล่าสุด
    joblib.dump(model, model_path2)
    joblib.dump(datetime.now(), date_path)
    print(f"บันทึกโมเดลและวันที่เทรนล่าสุดสำหรับ Product_code: {product_code}")
    
    return model

#===============================================predict_next_month=========================================
def predict_next_month(model, scaled_data, scaler):
    latest_data = scaled_data[-30:]  # ใช้ข้อมูลจาก 30 วันสุดท้าย
    latest_data = latest_data.reshape((1, latest_data.shape[0], 1))  # ปรับรูปแบบข้อมูล
    next_month_prediction = model.predict(latest_data)

    # เติมค่าให้เหมาะสมกับการ inverse transform
    next_month_full = np.zeros((1, scaled_data.shape[1]))
    next_month_full[:, 0] = next_month_prediction[:, 0]
    return scaler.inverse_transform(next_month_full)
    
#===============================================API=========================================
# Flask App
app = Flask(__name__)

@app.route('/', methods=['GET'])
def predict_sales():
    # โหลดข้อมูล
    df = load_data()

    # เตรียมข้อมูล
    X, y, scaler, df_prepared = prepare_data(df)

    # ปรับรูปร่างข้อมูลให้เหมาะสมกับ LSTM
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # แบ่งข้อมูลเป็น training และ testing set (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # ฝึกโมเดล
    model = train_lstm_model1(X_train, y_train)

    # ทำนายยอดขายในวันถัดไปโดยใช้ข้อมูล testing
    predicted_sales = model.predict(X_test)
    predicted_sales = scaler.inverse_transform(predicted_sales)

    # คำนวณ mse สำหรับ Testing Data
    mse = mean_squared_error(y_test, predicted_sales)
    mae = mean_absolute_error(y_test, predicted_sales)
    mape = mean_absolute_percentage_error(y_test, predicted_sales)
    # เตรียมข้อมูลเพิ่มเติม
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

            # สร้างเดือนถัดไป
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


    # รวมข้อมูลเพื่อส่งกลับ
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