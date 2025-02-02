from datetime import datetime, timedelta
from pycaret.regression import *
from flask import Flask, jsonify
import pandas as pd
import os
import sys
import joblib

# กำหนดเส้นทางให้เข้าถึงโมดูล
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'PJML')))

# นำเข้าโมดูลที่กำหนด
from Datafile.load_data import load_data
from Preprocess.preprocess_data import preprocess_data

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
        data['sale_date'] = pd.to_datetime(data['sale_date'], errors='coerce')
        data = data.dropna(subset=['sale_date'])
        
        print(f"Missing sale_date entries: {data['sale_date'].isna().sum()}")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, 'ModelPyCaret')
        os.makedirs(model_dir, exist_ok=True)

        model_path1 = os.path.join(model_dir, 'Pycaret_model.pkl')
        date_path = os.path.join(model_dir, 'last_trained_date1.pkl')

        # โหลดโมเดลหากมีอยู่
        if os.path.exists(model_path1):
            model = joblib.load(model_path1)
            print("📥 โหลดโมเดลจากไฟล์ที่เก็บไว้แล้ว")
        else:
            model = None

        # โหลดวันที่ฝึกล่าสุด
        if os.path.exists(date_path):
            last_trained_date = joblib.load(date_path)
        else:
            last_trained_date = datetime.min

        # ตรวจสอบว่าจำเป็นต้องเทรนใหม่หรือไม่
        if datetime.now() - last_trained_date < timedelta(days=30):
            print("⏳ ยังไม่ถึงเวลาเทรนใหม่ (ต้องรออย่างน้อย 30 วัน)")
            return jsonify({"message": "Model is still up-to-date."})
        
        print("🔄 ถึงเวลาเทรนโมเดลใหม่!")
        
        # PyCaret Setup
        s = setup(
            data=data,
            target='sales_amount',
            session_id=123,
            feature_selection=True,
            train_size=0.8,
            normalize=True,
            remove_outliers=True,
            verbose=False
        )
        
        # เลือกโมเดลที่ดีที่สุด
        best_model = compare_models()
        tuned_model = tune_model(best_model)
        
        # ทำนายค่าจากโมเดล
        prediction = predict_model(tuned_model, data=data)
        
        # บันทึกโมเดลและวันที่ฝึกล่าสุด
        joblib.dump(tuned_model, model_path1)
        print(f"💾 โมเดลถูกบันทึกไว้ที่ {model_path1}")
        
        joblib.dump(datetime.now(), date_path)
        print(f"📅 วันที่ฝึกโมเดลล่าสุดถูกบันทึกไว้ที่ {date_path}")
        
        # ดึงค่าพยากรณ์
        predicted_sales = prediction.loc[0, 'prediction_label']
        last_date = data['sale_date'].iloc[-1]
        predicted_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        
        # ดึงค่าประสิทธิภาพโมเดล
        metrics = pull()
        r2 = metrics.loc[0, "R2"]
        mae = metrics.loc[0, "MAE"]
        mape = metrics.loc[0, "MAPE"]
        
        return jsonify({
            "r2": r2,
            "mae": mae,
            "mape": mape,
            "best_model": str(best_model),
            "predicted_date": predicted_date,
            "predicted_sales": predicted_sales
        })
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='localhost', port=8885, debug=True)