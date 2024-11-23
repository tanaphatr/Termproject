import sys
import os

# เพิ่ม path ของโปรเจค
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'PJML')))

from flask import Flask, jsonify, request
import pandas as pd
import joblib
from Datafile.load_data import load_data
from Preprocess.preprocess_data import preprocess_data
from Trainmodel.main import train_model
from datetime import datetime, timedelta

def should_train_new_model():
    model_path = 'E:/Term project/PJ/PJML/Model'
    
    # ตรวจสอบว่าโมเดลถูกฝึกครั้งล่าสุดเมื่อไหร่
    model_time_filename = os.path.join(model_path, 'model_train_time.txt')
    
    if not os.path.exists(model_time_filename):
        # ถ้าไม่มีไฟล์เวลา เทรนโมเดลใหม่เลย
        return True
    
    with open(model_time_filename, 'r') as f:
        model_train_time = datetime.strptime(f.read().strip(), '%Y-%m-%d %H:%M:%S.%f')
    
    # ถ้าผ่านไป 30 วันจากการฝึกครั้งล่าสุด ให้ฝึกโมเดลใหม่
    if datetime.now() - model_train_time >= timedelta(days=30):
        return True
    
    return False

def load_trained_model():
    # พาธของไฟล์โมเดลที่บันทึกไว้
    model_path = 'E:/Term project/PJ/PJML/Model/model.pkl'
    
    # โหลดโมเดลที่บันทึกไว้
    model = joblib.load(model_path)
    
    # พาธของไฟล์ metrics ที่บันทึกไว้
    model_metrics_path = 'E:/Term project/PJ/PJML/Model/model_metrics.csv'
    
    # โหลด metrics ที่บันทึกไว้
    model_metrics = pd.read_csv(model_metrics_path)
    
    return model, model_metrics

app = Flask(__name__)

@app.route('/', methods=['GET'])
def predict_sales():
    # ตรวจสอบว่าเราต้องฝึกโมเดลใหม่หรือไม่
    if should_train_new_model():
        model, model_metrics, model_name = train_model()  # ฝึกโมเดลใหม่
    else:
        model, model_metrics = load_trained_model()  # โหลดโมเดลที่ฝึกเสร็จแล้ว

    # โหลดและประมวลผลข้อมูล
    data = load_data()
    processed_data = preprocess_data(data)
    
    # เลือกข้อมูลล่าสุด
    latest_data = processed_data.iloc[-1:].drop(columns=['sale_date', 'sales_amount'])

    # ทำนายยอดขาย
    predicted_sales = model.predict(latest_data)

    # ดึงวันที่จากข้อมูลล่าสุดและเพิ่มวัน
    predicted_date = data.iloc[-1]['sale_date'] + pd.DateOffset(days=1)

    # กรองข้อมูลที่ตรงกับ model_name
    model_name_cleaned = model_name.lower().replace(" ", "")
    filtered_model_metrics_name = model_metrics[model_metrics['Model'].str.lower().str.replace(" ", "") == model_name_cleaned]
    model_metrics_filtered = filtered_model_metrics_name[['MAE', 'MAPE', 'RMSE', 'R2']]

    # แปลง model_metrics ให้เป็น dictionary
    model_metrics_dict = model_metrics_filtered.to_dict(orient='records') if isinstance(model_metrics, pd.DataFrame) else model_metrics
    
    return jsonify({
        'predicted_sales': predicted_sales[0],
        'predicted_date': predicted_date.strftime("%Y-%m-%d"),
        'model_name': model_name,
        'model_metrics': model_metrics_dict
    })


if __name__ == '__main__':
    app.run(host='localhost', port=8887, debug=True)
