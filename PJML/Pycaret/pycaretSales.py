import os
import sys
<<<<<<< HEAD
from flask import Flask, jsonify
from pycaret.regression import *
import pandas as pd
from datetime import datetime, timedelta
=======
from pycaret.regression import *
>>>>>>> cc71bb4222b7d2f19c57814542fa4522a7b8d353

# กำหนดเส้นทางให้เข้าถึงโมดูล
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'PJML')))

# นำเข้าโมดูลที่กำหนด
from Datafile.load_data import load_data
from Preprocess.preprocess_data import preprocess_data

<<<<<<< HEAD
# กำหนดฟีเจอร์และตัวแปรเป้าหมาย
FEATURES = [
    'weather_Mostly Sunny', 'weather_Partly Cloudy', 'weather_Scattered Shower',
    'event_Normal Day',
]
TARGET = 'sales_amount'

# สร้าง Flask Application
app = Flask(__name__)

@app.route('/', methods=['GET'])
def forecast_sales():
    try:
        # โหลดข้อมูล
        data = load_data()

        # ทำการ preprocess ข้อมูล
        data = preprocess_data(data)

        # ตรวจสอบและจัดการค่า NaT ใน sale_date
        data['sale_date'] = pd.to_datetime(data['sale_date'], errors='coerce')  # แปลงคอลัมน์ sale_date เป็น datetime และทำให้ค่าที่แปลงไม่ได้เป็น NaT
        data = data.dropna(subset=['sale_date'])  # ลบแถวที่มี NaT ในคอลัมน์ sale_date

        # ตรวจสอบว่าฟีเจอร์ที่ต้องการมีในข้อมูลหรือไม่
        for feature in FEATURES + [TARGET]:
            if feature not in data.columns:
                return jsonify({"error": f"Missing required feature: {feature}"}), 400

        # สร้าง PyCaret Setup
        reg = setup(
            data=data,
            target=TARGET,
            train_size=0.8,
            session_id=123,
            verbose=False
        )

        # เปรียบเทียบโมเดลและเลือกโมเดลที่ดีที่สุด
        best_model = compare_models()

        # ทดสอบโมเดลที่เลือกและเพิ่มการปรับแต่ง hyperparameters (รวมถึง epochs ถ้าเป็นไปได้)
        tuned_model = tune_model(best_model)

        # ทำนายด้วยโมเดลที่ได้รับการปรับแต่งแล้ว
        prediction = predict_model(tuned_model, data=data)

        # ดึงข้อมูลการพยากรณ์ตัวอย่าง (บรรทัดแรก)
        predicted_sales = prediction.loc[0, 'prediction_label']  # 'prediction_label' เป็นผลลัพธ์ที่ PyCaret สร้างหลังพยากรณ์

        # คำนวณวันที่ถัดไป
        last_date = data['sale_date'].iloc[-1]  # วันสุดท้ายในข้อมูล (ใช้ sale_date แทน index)
        if isinstance(last_date, pd.Timestamp):  # หาก last_date เป็น Timestamp
            predicted_date = last_date + timedelta(days=1)  # เพิ่ม 1 วัน
        else:
            # ถ้าไม่ใช่ Timestamp ให้แปลงเป็น datetime ก่อน
            last_date = pd.to_datetime(last_date)
            predicted_date = last_date + timedelta(days=1)

        predicted_date = predicted_date.strftime("%Y-%m-%d")  # แปลงให้เป็น string format

        # ดึงค่า MAE, MSE, RMSE และ R²
        r2 = pull().loc[0, "R2"]
        mae = pull().loc[0, "MAE"]
        mape = pull().loc[0, "MAPE"]

        # ส่งผลลัพธ์ผ่าน API
        return jsonify({
            "r2": r2,
            "mae": mae,
            "mape": mape,
            "best_model": str(best_model),
            "predicted_date": predicted_date,
            "predicted_sales": predicted_sales
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='localhost', port=8885, debug=True)
=======
# โหลดและประมวลผลข้อมูล
df = load_data()
df = preprocess_data(df)

# สร้างโมเดล PyCaret พร้อมการเลือกฟีเจอร์อัตโนมัติ
def Model(df):

    # ลบข้อมูลที่ไม่เกี่ยวข้อง
    if 'sale_date' in df.columns:
        df = df.drop(columns=['sale_date'])

    # ตั้งค่าการเตรียมข้อมูลใน PyCaret
    setup(data=df, target='sales_amount', session_id=123, 
          feature_selection=True, 
          categorical_features=df.select_dtypes(include=['object']).columns.tolist(),
          numeric_imputation='mean',
          train_size=0.8)
    
    # เปรียบเทียบโมเดล
    best_model = compare_models()
    
    # ปรับจูนโมเดลที่ดีที่สุด
    tuned_model = tune_model(best_model)
    
    # บันทึกโมเดล
    save_model(tuned_model, 'sales_prediction_with_fs_model')
    print("โมเดลถูกบันทึกเรียบร้อยแล้ว: sales_prediction_with_fs_model.pkl")
    
    return tuned_model

# เรียกใช้ฟังก์ชันสร้างโมเดล
sales_model_with_fs = Model(df)
>>>>>>> cc71bb4222b7d2f19c57814542fa4522a7b8d353
