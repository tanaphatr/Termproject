import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, jsonify
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense  # type: ignore
from sklearn.model_selection import train_test_split

# เพิ่ม path ของโปรเจค
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'PJML')))

from Datafile.load_data import load_data
#===============================================เตรียมข้อมูล=========================================
def prepare_data(df):
    # เติมค่าขาดหายไปในคอลัมน์เป้าหมาย
    df['sales_amount'] = df['sales_amount'].fillna(df['sales_amount'].mean())

    # แปลงวันที่จาก พ.ศ. เป็น ค.ศ.
    if 'sale_date' in df.columns:
        df['sale_date'] = df['sale_date'].astype(str)  # แปลงเป็น string
        df['sale_date'] = df['sale_date'].apply(lambda x: str(int(x[:4]) - 543) + x[4:] if len(x) == 10 else x)

        # แปลงวันที่ให้เป็น datetime และเรียงลำดับ
        df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
        df = df.sort_values('sale_date')

    # สเกลข้อมูลให้อยู่ในช่วง [0, 1]
    scaler = MinMaxScaler()
    df['sales_amount_scaled'] = scaler.fit_transform(df[['sales_amount']])

    # เตรียมข้อมูลสำหรับ LSTM
    sequence_length = 10  # จำนวนวันใน sequence
    X, y = [], []
    for i in range(sequence_length, len(df)):
        X.append(df['sales_amount_scaled'].iloc[i-sequence_length:i].values)
        y.append(df['sales_amount_scaled'].iloc[i])

    # บันทึกคอลัมน์ sale_date เป็นไฟล์ CSV
    df[['sale_date']].to_csv('sale_dates.csv', index=False)

    return np.array(X), np.array(y), scaler, df
#===============================================เทรน=========================================
def train_lstm_model(X, y):
    # สร้างโมเดล LSTM
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=32, verbose=1)
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

# Flask App
app = Flask(__name__)
#===============================================API=========================================
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
    model = train_lstm_model(X_train, y_train)

    # ทำนายยอดขายในวันถัดไปโดยใช้ข้อมูล testing
    predicted_sales = model.predict(X_test)
    predicted_sales = scaler.inverse_transform(predicted_sales)

    # แปลงค่า y_test กลับจากการสเกลเป็นยอดขายจริง
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # สร้างกราฟเปรียบเทียบผลลัพธ์
    plt.figure(figsize=(10, 6))
    plt.plot(df_prepared['sale_date'].iloc[-len(y_test):], y_test_actual, label="Actual Sales", color='blue')
    plt.plot(df_prepared['sale_date'].iloc[-len(y_test):], predicted_sales, label="Predicted Sales", color='red')
    plt.xlabel('Date')
    plt.ylabel('Sales Amount')
    plt.title('Actual vs Predicted Sales')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return jsonify({
        'predicted_sales': float(predicted_sales[-1][0]),
        'predicted_date': df_prepared['sale_date'].iloc[-1] + pd.DateOffset(days=1),
        'model_name': "LSTM"
    })

if __name__ == '__main__':
    app.run(host='localhost', port=8885, debug=True)
