import os
import sys
import pandas as pd
import joblib
from flask import Flask, jsonify
from pycaret.regression import setup, compare_models, pull, save_model, load_model, predict_model

# เพิ่ม path ของโปรเจค
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'PJML')))

from Datafile.load_data import load_data
from Preprocess.preprocess_data import preprocess_data

#===============================================เตรียมข้อมูล=========================================
def prepare_data(df):
    df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
    df = df.dropna(subset=['sale_date'])

    # แปลงค่าฟีเจอร์ weather และ event ให้อยู่ในรูปของตัวเลข
    weather_columns = ['weather_Mostly Sunny', 'weather_Partly Cloudy', 'weather_Scattered Shower']
    event_column = ['event_Normal Day']

    df[weather_columns] = df[weather_columns].astype(int)
    df[event_column] = df[event_column].astype(int)

    # เติมค่าขาดหายไปในคอลัมน์ Temperature (ถ้ามี)
    if 'Temperature' in df.columns:
        df['Temperature'] = df['Temperature'].fillna(df['Temperature'].mean())

    # เพิ่มฟีเจอร์ปี เดือน วัน
    df['Year'] = df['sale_date'].dt.year
    df['Month'] = df['sale_date'].dt.month
    df['Day'] = df['sale_date'].dt.day

    return df

#===============================================เทรน=========================================
def train_pycaret_model(df):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'ModelPyCaret')
    model_path = os.path.join(model_dir, 'pycaret_model')

    # ตรวจสอบและสร้างโฟลเดอร์เก็บโมเดล
    os.makedirs(model_dir, exist_ok=True)

    # เช็คว่าโมเดลมีอยู่หรือไม่
    if os.path.exists(model_path + '.pkl'):
        print("โหลดโมเดลจากไฟล์ที่มีอยู่")
        return load_model(model_path)

    # กำหนดฟีเจอร์และตัวแปรเป้าหมาย
    target = 'sales_amount'
    features = ['Temperature', 'Year', 'Month', 'Day', 'weather_Mostly Sunny', 'weather_Partly Cloudy',
                'weather_Scattered Shower', 'event_Normal Day']

    df = df[features + [target]].dropna()

    # ตั้งค่าด้วย PyCaret
    reg_setup = setup(data=df, target=target, session_id=123, fold=5)

    # เลือกโมเดลที่ดีที่สุด
    best_model = compare_models()

    # ประเมินผลโมเดล
    model_results = pull()
    
    # คำนวณ R2, MAE, MAPE
    r2_score = model_results.loc[best_model.iloc[0], 'R2']
    mae_score = model_results.loc[best_model.iloc[0], 'MAE']
    mape_score = model_results.loc[best_model.iloc[0], 'MAPE']

    print(f"R2 Score: {r2_score}")
    print(f"MAE Score: {mae_score}")
    print(f"MAPE Score: {mape_score}")

    # บันทึกโมเดล
    save_model(best_model, model_path)
    print("โมเดล PyCaret ถูกบันทึกแล้ว")

    return best_model

#===============================================API=========================================
app = Flask(__name__)

@app.route('/', methods=['GET'])
def predict_sales_api():
    df = load_data()

    # ส่ง df ไปยังฟังก์ชัน preprocess_data
    df_preprocessed = preprocess_data(df)

    # เตรียมข้อมูล
    df_prepared = prepare_data(df_preprocessed)

    # เทรนโมเดล
    model = train_pycaret_model(df_prepared)

    # ทำนายยอดขายในวันถัดไป
    last_row = df_prepared.iloc[[-1]].copy()
    last_row['sale_date'] += pd.DateOffset(days=1)

    prediction = predict_model(model, data=last_row)

    # ใช้คอลัมน์ 'prediction_label' ที่ได้จากการทำนาย
    predicted_sales = prediction['prediction_label'].iloc[0]
    predicted_date = last_row['sale_date'].iloc[0]

    return jsonify({
        'predicted_sales': float(predicted_sales),
        'predicted_date': str(predicted_date),
        'model_name': "PyCaret",
    })


if __name__ == '__main__':
    app.run(host='localhost', port=8885, debug=True)
