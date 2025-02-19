import sys
import os

# เพิ่ม path ของโปรเจค
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'PJML')))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from Datafile.load_data import load_data
from pycaret.regression import *

def add_lag_features(df, columns=['sales_amount'], lags=[1, 7, 30]):
    """สร้าง lag features"""
    for col in columns:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df

def add_rolling_statistics(df, columns=['sales_amount'], windows=[30, 60]):
    """สร้าง rolling statistics features"""
    for col in columns:
        for window in windows:
            df[f'{col}_ma_{window}'] = df[col].rolling(window=window).mean()
            df[f'{col}_std_{window}'] = df[col].rolling(window=window).std()
    return df

def add_cyclical_features(df, date_column='sale_date'):
    """สร้าง cyclical features"""
    df['month_sin'] = np.sin(2 * np.pi * df[date_column].dt.month/12)
    df['month_cos'] = np.cos(2 * np.pi * df[date_column].dt.month/12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df[date_column].dt.dayofweek/7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df[date_column].dt.dayofweek/7)
    return df

def add_noise(df, columns=['sales_amount'], noise_factor=0.01):
    """เพิ่ม noise ให้กับข้อมูล"""
    df_noise = df.copy()
    for col in columns:
        noise = np.random.normal(0, noise_factor * df[col].std(), size=len(df))
        df_noise[col] = df[col] + noise
    return df_noise

def jitter_data(df, noise_level=0.02):
    """เพิ่ม jitter ให้กับข้อมูลยอดขาย"""
    df_jitter = df.copy()
    df_jitter['sales_amount_jitter'] = df['sales_amount'] + np.random.normal(0, df['sales_amount'].std() * noise_level, len(df))
    return df_jitter

def time_shift(df, shift_days=1):
    """สร้างข้อมูลด้วยการเลื่อนเวลา"""
    df_shifted = df.copy()
    df_shifted['sale_date'] = df_shifted['sale_date'] + pd.Timedelta(days=shift_days)
    return df_shifted

def scale_features(df, columns=['sales_amount'], scale_factor=1.1):
    """ปรับสเกลข้อมูล"""
    df_scaled = df.copy()
    for col in columns:
        df_scaled[col] = df[col] * scale_factor
    return df_scaled

def train_modeltest():
    # โหลดข้อมูล
    df = load_data()

    # เติมค่าขาดหายไป
    df['sales_amount'] = df['sales_amount'].fillna(df['sales_amount'].mean())

    # แปลงวันที่
    if 'sale_date' in df.columns:
        df['sale_date'] = df['sale_date'].apply(
            lambda x: pd.to_datetime(str(int(x[:4]) - 543) + x[4:]) if isinstance(x, str) else pd.NaT
        )

    # เพิ่ม features ใหม่
    df = add_lag_features(df)
    df = add_rolling_statistics(df)
    df = add_cyclical_features(df)

    # Data Augmentation
    df_noise = add_noise(df)
    df_shifted = time_shift(df)
    df_scaled = scale_features(df)

    # รวมข้อมูลที่เพิ่มขึ้น
    df_augmented = pd.concat([df, df_noise, df_shifted, df_scaled], ignore_index=True)

    # จัดการ categorical features
    categorical_features = df_augmented.select_dtypes(include=['object']).columns.tolist()
    for column in categorical_features:
        df_augmented[column] = df_augmented[column].str.replace(' ', '_', regex=True)

    categorical = ['event', 'weather']

    # ตั้งค่า PyCaret
    setup(
        data=df_augmented,
        target='sales_amount',
        session_id=123,
        categorical_features=categorical,
        feature_selection=True,
        normalize=True,
        remove_outliers=True,
        train_size=0.8,
    )

    # เปรียบเทียบและเลือกโมเดลที่ดีที่สุด
    best_model = compare_models()
    final_model = finalize_model(best_model)

    # บันทึกข้อมูลที่ใช้ในการเทรน
    X_train = get_config('X_train')
    X_train.to_csv('transformed_training_data.csv', index=False)

    model_metrics = pull()
    model_name = best_model.__class__.__name__

    return final_model, model_metrics, model_name, df_augmented

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