import sys
import os

# เพิ่ม path ของโปรเจค
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'PJML')))

from flask import Flask, jsonify, request
import pandas as pd
from Datafile.load_data import load_data
from Preprocess.preprocess_data import preprocess_data
from Trainmodel.main import train_model

app = Flask(__name__)

@app.route('/', methods=['GET'])
def predict_sales():
    # โหลดโมเดลที่ฝึกเสร็จแล้ว
    model, model_metrics, model_name = train_model()  # ดึงข้อมูลจาก train_model

    # โหลดและประมวลผลข้อมูล
    data = load_data()
    processed_data = preprocess_data(data)
    
    # เลือกข้อมูลล่าสุด (ตัวอย่าง: ใช้ข้อมูลล่าสุดจาก processed_data)
    latest_data = processed_data.iloc[-1:].drop(columns=['sale_date', 'sales_amount'])

    # ทำนายยอดขาย
    predicted_sales = model.predict(latest_data)

    # ดึงวันที่จากข้อมูลล่าสุดและเพิ่มวัน
    predicted_date = data.iloc[-1]['sale_date'] + pd.DateOffset(days=1)

    # ลบช่องว่างและแปลงเป็นตัวพิมพ์เล็กก่อนทำการกรอง
    model_name_cleaned = model_name.lower().replace(" ", "")

    # กรองข้อมูลที่ตรงกับ model_name โดยไม่สนใจพิมพ์เล็กพิมพ์ใหญ่และช่องว่าง
    filtered_model_metrics_name = model_metrics[model_metrics['Model'].str.lower().str.replace(" ", "") == model_name_cleaned]

    # สมมติว่า model_metrics เป็น DataFrame ที่มีข้อมูลทั้งหมด
    model_metrics_filtered = filtered_model_metrics_name[['MAE', 'MAPE', 'RMSE', 'R2']]

    # แปลง model_metrics ให้เป็น dictionary ถ้าจำเป็น
    model_metrics_dict = model_metrics_filtered.to_dict(orient='records') if isinstance(model_metrics, pd.DataFrame) else model_metrics
    
    return jsonify({
        'predicted_sales': predicted_sales[0],
        'predicted_date': predicted_date.strftime("%Y-%m-%d"),
        'model_name': model_name,  # ส่งชื่อโมเดลที่เลือก
        'model_metrics': model_metrics_dict  # ส่ง model_metrics ที่กรองเฉพาะ model_name
    })


if __name__ == '__main__':
    app.run(host='localhost', port=8887, debug=True)
