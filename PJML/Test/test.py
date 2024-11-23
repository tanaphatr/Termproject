import sys
import os

# เพิ่ม path ของโปรเจค
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'PJML')))

import pandas as pd
from Datafile.load_data import load_data
from pycaret.regression import *

def train_modeltest():
    # โหลดข้อมูล
    df = load_data()

    # เติมค่าขาดหายไปในคอลัมน์เป้าหมาย
    df['sales_amount'] = df['sales_amount'].fillna(df['sales_amount'].mean())

    # ลบข้อมูลที่ไม่เกี่ยวข้อง
    if 'sale_date' in df.columns:
        # แปลงจาก พ.ศ. เป็น ค.ศ. ก่อนทำการแปลงเป็น datetime
        df['sale_date'] = df['sale_date'].apply(
            lambda x: pd.to_datetime(str(int(x[:4]) - 543) + x[4:]) if isinstance(x, str) else pd.NaT
        )

    # เลือกคอลัมน์ที่เป็น object (Categorical Features)
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    print(categorical_features)

    # เติม _ ในส่วนที่เป็นช่องว่างใน categorical features
    for column in categorical_features:
        df[column] = df[column].str.replace(' ', '_', regex=True)

    categorical = ['event', 'festival', 'weather', 'Back_to_School_Period']
    # ตั้งค่าการเตรียมข้อมูลใน PyCaret
    setup(
        data=df,
        target='sales_amount',
        session_id=123,
        categorical_features=categorical,
        feature_selection=True,
        train_size=0.8,
    )

    # เปรียบเทียบโมเดลต่างๆ
    best_model = compare_models()

    # ฝึกโมเดลที่ดีที่สุด
    final_model = finalize_model(best_model)

    # ดึงข้อมูลที่แปลงแล้วที่ใช้ในการฝึกโมเดล
    X_train = get_config('X_train')

    # บันทึกข้อมูลที่ใช้ในการฝึกโมเดลลงในไฟล์ CSV
    X_train.to_csv('transformed_training_data.csv', index=False)

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
    lasttestdata = train_data.iloc[-1:].copy()  # คัดลอกข้อมูลมาใช้โดยไม่กระทบกับต้นฉบับ

    # ลบคอลัมน์ 'sales_amount' (target) ออกจากข้อมูลที่ใช้ทำนาย
    lasttestdata = lasttestdata.drop(columns=['sales_amount'])

    # ทำนายยอดขาย
    predicted_sales = model.predict(lasttestdata)

    # แปลง predicted_sales ให้เป็น float ก่อนส่งกลับ
    predicted_sales = float(predicted_sales[0])

    # ดึงวันที่จากข้อมูลล่าสุดและเพิ่มวัน
    predicted_date = data.iloc[-1]['sale_date'] + pd.DateOffset(days=1)

    # ลบช่องว่างและแปลงเป็นตัวพิมพ์เล็กก่อนทำการกรอง
    model_name_cleaned = model_name.lower().replace(" ", "")

    # กรองข้อมูลที่ตรงกับ model_name โดยไม่สนใจพิมพ์เล็กพิมพ์ใหญ่และช่องว่าง
    filtered_model_metrics_name = model_metrics[model_metrics['Model'].str.lower().str.replace(" ", "") == model_name_cleaned]

    # สมมติว่า model_metrics เป็น DataFrame ที่มีข้อมูลทั้งหมด
    model_metrics_filtered = filtered_model_metrics_name[['MAE', 'MAPE', 'RMSE', 'R2']]

    # แปลง model_metrics ให้เป็น dictionary ถ้าจำเป็น
    model_metrics_filtered = model_metrics_filtered.applymap(float)  # แปลงค่าทั้งหมดเป็น float

    model_metrics_dict = model_metrics_filtered.to_dict(orient='records') if isinstance(model_metrics, pd.DataFrame) else model_metrics
    
    return jsonify({
        'predicted_sales': predicted_sales,
        'predicted_date': predicted_date.strftime("%Y-%m-%d"),
        'model_name': model_name,  # ส่งชื่อโมเดลที่เลือก
        'model_metrics': model_metrics_dict  # ส่ง model_metrics ที่กรองเฉพาะ model_name
    })




if __name__ == '__main__':
    app.run(host='localhost', port=8885, debug=True)
