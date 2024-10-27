import sys
import os

# เพิ่ม path ของโปรเจค
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'PJML')))

from flask import Flask, jsonify, request
import pandas as pd
from Datafile.load_data import load_data
from Preprocess.preprocess_data import preprocess_data
from Trainmodel.main import train_model  # นำเข้า train_model จาก main.py

app = Flask(__name__)

@app.route('/', methods=['GET'])
def predict_sales():
    # ฝึกโมเดลทุกครั้งที่เรียกใช้ API
    linear_regression_model = train_model()
    
    # โหลดและประมวลผลข้อมูล
    data = load_data()
    processed_data = preprocess_data(data)
    
    # Convert sale_date to datetime and extract relevant features
    data['sale_date'] = pd.to_datetime(data['sale_date'])
    data['year'] = data['sale_date'].dt.year
    data['month'] = data['sale_date'].dt.month
    data['day'] = data['sale_date'].dt.day
    data['day_of_year'] = data['sale_date'].dt.dayofyear

    # สร้าง latest_data
    latest_data = data.iloc[-1][[
        'year', 
        'month', 
        'day', 
        'day_of_year',
        'event', 
        'day_of_week', 
        'festival', 
        'weather', 
        'Back_to_School_Period',
        'Seasonal'
    ]].values.reshape(1, -1)

    # ทำนายยอดขายวันถัดไป
    predicted_sales = linear_regression_model.predict(latest_data)

    # ดึงวันที่จากข้อมูล
    predicted_date = data.iloc[-1]['sale_date'] + pd.DateOffset(days=1)

    # คุณสามารถส่งยอดขายจริงจาก request หรือค่าตัวอย่างก็ได้
    actual_sales = request.args.get('actual_sales', type=float, default=9379)

    # คำนวณค่าผิดพลาดและเปอร์เซ็นต์ความผิดพลาด
    error = predicted_sales[0] - actual_sales
    percentage_error = (error / actual_sales) * 100

    # ส่งผลลัพธ์กลับในรูปแบบ JSON
    return jsonify({
        'predicted_sales': predicted_sales[0],
        'predicted_date': predicted_date.strftime("%Y-%m-%d"),
        'error': error,
        'percentage_error': percentage_error
    })

if __name__ == '__main__':
    app.run(host='localhost', port=8887 , debug=True)
