import sys
import os

# เพิ่ม path ของโปรเจค
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'PJML')))

import pandas as pd
from Datafile.load_data import load_data
from Preprocess.preprocess_data import preprocess_data
from statsmodels.tsa.arima.model import ARIMA
from flask import Flask, jsonify
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import numpy as np

app = Flask(__name__)

# ฟังก์ชันสำหรับฝึกโมเดล ARIMA
def train_arima_model():
    # โหลดข้อมูล
    data = load_data()

    # เตรียมข้อมูล (เช่น เลือกแค่คอลัมน์ที่จำเป็นและจัดการกับค่าว่าง)
    df = preprocess_data(data)

    # สมมติว่า 'sale_date' คือวันที่ และ 'sales_amount' คือยอดขาย
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    df.set_index('sale_date', inplace=True)

    # ใช้ข้อมูลยอดขาย (sales_amount) สำหรับฝึกโมเดล ARIMA
    sales_data = df['sales_amount']

    # สร้างโมเดล ARIMA (พารามิเตอร์ p, d, q ที่เลือกมาจากการทดสอบหรือการวิจัย)
    model = ARIMA(sales_data, order=(10, 1, 0))  # p=5, d=1, q=0 (คุณสามารถปรับได้)
    model_fit = model.fit()

    # ทำนายยอดขายในอนาคต 1 วัน (จากวันที่ล่าสุดในข้อมูล)
    forecast = model_fit.forecast(steps=1)  # ทำนายแค่วันถัดไปจากข้อมูลล่าสุด

    # คำนวณ MAE, MAPE, R²
    # ใช้ข้อมูลยอดขายจริงจากวันที่ล่าสุดถึงวันที่ที่ทำนาย
    y_true = sales_data[-1:]  # ยอดขายจริงจากข้อมูล (ในที่นี้ใช้แค่ข้อมูลล่าสุด)
    y_pred = forecast  # ยอดขายที่ทำนาย

    mae = mean_absolute_error(y_true, y_pred)  # MAE
    mape = mean_absolute_percentage_error(y_true, y_pred)  # MAPE

    # เพิ่มวัน +1
    latest_date = df.index.max()
    latest_date_plus_one = latest_date + pd.Timedelta(days=1)

    return forecast, mae, mape, latest_date_plus_one

# Route สำหรับการทำนายยอดขาย
@app.route('/', methods=['GET'])
def predict_sales():
    forecast, mae, mape, latest_date_plus_one = train_arima_model()
    return jsonify({
        'forecast': forecast.tolist(),  # ส่งผลการทำนายเป็น JSON
        'mae': mae,  # ส่ง MAE
        'mape': mape,  # ส่ง MAPE
        'latest_date': latest_date_plus_one.strftime('%Y-%m-%d')  # ส่งวันที่ล่าสุดที่เพิ่มวันแล้ว
    })

if __name__ == '__main__':
    app.run(host='localhost', port=8887, debug=True)
