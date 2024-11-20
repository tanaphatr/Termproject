import sys
import os

# เพิ่ม path ของโปรเจค
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'PJML')))

import pandas as pd
from Datafile.load_data import load_data
from pycaret.regression import setup, compare_models, finalize_model, pull

def train_modeltest():
    # โหลดข้อมูล
    df = load_data()

    # เติมค่าขาดหายไปในคอลัมน์เป้าหมาย
    df['sales_amount'] = df['sales_amount'].fillna(df['sales_amount'].mean())
    
    # ลบข้อมูลที่ไม่เกี่ยวข้อง
    if 'sale_date' in df.columns:
        # แปลงจาก พ.ศ. เป็น ค.ศ. ก่อนทำการแปลงเป็น datetime
        df['sale_date'] = df['sale_date'].apply(lambda x: pd.to_datetime(str(int(x[:4]) - 543) + x[4:]) if isinstance(x, str) else pd.NaT)

    # ตั้งค่าการเตรียมข้อมูลใน PyCaret
    setup(data=df, target='sales_amount', session_id=123, 
          feature_selection=True, 
          categorical_features=df.select_dtypes(include=['object']).columns.tolist(),
          numeric_imputation='mean',
          train_size=0.8)

    # เปรียบเทียบโมเดลต่างๆ
    best_model = compare_models()

    # ฝึกโมเดลที่ดีที่สุด
    final_model = finalize_model(best_model)

    # ดึงผลลัพธ์ metrics
    model_metrics = pull()
    model_name = best_model.__class__.__name__

    return final_model, model_metrics, model_name, df

from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def predict_sales():
    # เรียกฟังก์ชัน train_modeltest เพื่อฝึกโมเดลและดึงค่าที่เกี่ยวข้อง
    model, model_metrics, model_name, train_data = train_modeltest()

    # โหลดข้อมูล
    data = load_data()

    # ใช้ข้อมูลจาก train_data สำหรับการทำนาย (หรือใช้ข้อมูลล่าสุดจาก train_data)
    lasttestdata = train_data.iloc[-1:].drop(columns=['sales_amount'], errors='ignore')

    # ปรับชื่อคอลัมน์ให้ตรงกับฟีเจอร์ที่ฝึกโมเดล
    lasttestdata.columns = [col.replace(" ", "_") for col in lasttestdata.columns]

    # ตรวจสอบว่ามีฟีเจอร์ใดขาดไปจากข้อมูลฝึกหรือไม่
    missing_cols = [col for col in model.feature_names_in_ if col not in lasttestdata.columns]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        # เติมฟีเจอร์ที่หายไปด้วยค่า 0 หรือค่าคงที่ที่เหมาะสม
        for col in missing_cols:
            lasttestdata[col] = 0

    # เติมค่าที่หายไปใน lasttestdata
    lasttestdata = lasttestdata.fillna(lasttestdata.mean())

    # ปรับชื่อฟีเจอร์ให้ตรงกันและเติมคอลัมน์ที่ขาดหายไป
    lasttestdata = lasttestdata.reindex(columns=model.feature_names_in_, fill_value=0)

    # ทำนายยอดขาย
    predicted_sales = model.predict(lasttestdata)

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
