import sys
import os

# เพิ่ม path ของโปรเจค
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'PJML')))

import pandas as pd
from Datafile.load_data import load_data
from tpot import TPOTRegressor

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

    # เติม _ ในส่วนที่เป็นช่องว่างใน categorical features
    for column in categorical_features:
        df[column] = df[column].str.replace(' ', '_', regex=True)

    # แยกข้อมูลเป็น X (Features) และ y (Target)
    X = df.drop(columns=['sales_amount'])
    y = df['sales_amount']

    # ใช้ TPOT ในการฝึกโมเดล
    tpot = TPOTRegressor(verbosity=2, generations=5, population_size=20)
    tpot.fit(X, y)

    # ดึงโมเดลที่ดีที่สุด
    best_model = tpot.fitted_pipeline_

    # บันทึกโมเดลที่ดีที่สุด
    tpot.export('best_model_pipeline.py')

    # ดึงผลลัพธ์ metrics (กรณีนี้จะใช้คะแนน R2)
    model_metrics = {'R2': best_model.score(X, y)}

    # ชื่อโมเดลที่ดีที่สุด
    model_name = "TPOT Regressor"

    return best_model, model_metrics, model_name, df

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

    # ดึงวันที่จากข้อมูลล่าสุดและเพิ่มวัน
    predicted_date = pd.Timestamp(data.iloc[-1]['sale_date']) + pd.DateOffset(days=1)

    # ส่งผลลัพธ์กลับไปยังผู้ใช้
    return jsonify({
        'predicted_sales': predicted_sales[0],
        'predicted_date': predicted_date.strftime("%Y-%m-%d"),
        'model_name': model_name,
        'model_metrics': model_metrics
    })


if __name__ == '__main__':
    app.run(host='localhost', port=8887, debug=True)
