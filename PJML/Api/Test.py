import sys
import os

# เพิ่ม path ของโปรเจค
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'PJML')))

from flask import Flask, jsonify
import pandas as pd
from Datafile.load_data import load_data
from Trainmodel.test import train_modeltest
from pycaret.regression import setup

app = Flask(__name__)

@app.route('/', methods=['GET'])
def predict_sales():
    # เรียกฟังก์ชัน train_modeltest เพื่อฝึกโมเดลและดึงค่าที่เกี่ยวข้อง
    model, model_metrics, model_name = train_modeltest()

    # โหลดข้อมูล
    data = load_data()

    # เตรียมข้อมูลล่าสุด (แถวสุดท้าย) สำหรับการทำนาย
    latest_data = data.iloc[-1:].drop(columns=['sales_amount'], errors='ignore')

    # ตั้งค่าการเตรียมข้อมูลใน PyCaret (จะมีการ One-Hot Encoding และเติมค่าขาดหาย)
    setup(data=data, target='sales_amount', session_id=123, categorical_features=data.select_dtypes(include=['object']).columns.tolist(), numeric_imputation='mean')

    # ปรับชื่อฟีเจอร์ใน latest_data ให้ตรงกับฟีเจอร์ที่โมเดลฝึก
    latest_data = latest_data.reindex(columns=model.feature_names_in_, fill_value=0)
    
    # ทำนายยอดขาย
    predicted_sales = model.predict(latest_data)

    # ดึงวันที่จากข้อมูลล่าสุดและเพิ่มวัน
    predicted_date = pd.Timestamp(data.iloc[-1]['sale_date']) + pd.DateOffset(days=1)

    # ทำความสะอาดชื่อโมเดลเพื่อนำไปกรองข้อมูล
    model_name_cleaned = model_name.lower().replace(" ", "")
    filtered_model_metrics = model_metrics[model_metrics['Model'].str.lower().str.replace(" ", "") == model_name_cleaned]

    # เลือกเฉพาะค่าที่เกี่ยวข้อง
    if not filtered_model_metrics.empty:
        model_metrics_filtered = filtered_model_metrics[['MAE', 'MAPE', 'RMSE', 'R2']].to_dict(orient='records')
    else:
        model_metrics_filtered = {}

    # ส่งผลลัพธ์กลับไปยังผู้ใช้
    return jsonify({
        'predicted_sales': predicted_sales[0],
        'predicted_date': predicted_date.strftime("%Y-%m-%d"),
        'model_name': model_name,
        'model_metrics': model_metrics_filtered
    })

if __name__ == '__main__':
    app.run(host='localhost', port=8887, debug=True)
