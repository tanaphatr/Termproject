import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from flask import Flask, jsonify
from pycaret.regression import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Datafile.load_data import load_dataps
from Preprocess.preprocess_data import preprocess_dataps

# รหัสสินค้าที่ต้องการทำนาย
product_codes = ["A1001", "A1002", "A1004", "A1034", "B1002", "B1003", "D1003"]
# product_codes = ["A1001", "A1002", "A1004", "A1034", "B1002", "B1003", "D1003"]

def add_time_features(df):
    """สร้าง time features"""
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    df['year'] = df['Date'].dt.year
    df['day_of_year'] = df['Date'].dt.dayofyear
    
    # Add cyclical encoding for seasonal features
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    return df

def add_new_features(df):
    """สร้าง features เพิ่มเติม"""
    df['prev_day_diff'] = df['Quantity'] - df['Quantity'].shift(1)
    df['rolling_avg_60'] = df['Quantity'].rolling(window=60).mean()
    return df

def add_lag_features(df, lags=[1, 2, 3, 7, 14, 30]):
    """สร้าง lag features"""
    for lag in lags:
        df[f'sales_lag_{lag}'] = df['Quantity'].shift(lag)
    return df

def add_rolling_features(df):
    """สร้าง rolling statistics features"""
    windows = [7, 14, 30]
    for window in windows:
        df[f'sales_ma_{window}'] = df['Quantity'].rolling(window=window).mean()
        df[f'sales_std_{window}'] = df['Quantity'].rolling(window=window).std()
        df[f'sales_min_{window}'] = df['Quantity'].rolling(window=window).min()
        df[f'sales_max_{window}'] = df['Quantity'].rolling(window=window).max()
    return df

def add_noise(df, noise_factor=0.03):
    """เพิ่ม noise ให้กับข้อมูล"""
    df_noise = df.copy()
    noise = np.random.normal(0, noise_factor * df['Quantity'].std(), size=len(df))
    df_noise['Quantity'] = np.clip(df_noise['Quantity'] + noise, 0, None)  # ป้องกันค่าติดลบ
    return df_noise

def time_shift(df, shifts=[-2, -1, 1, 2]):
    """สร้างข้อมูลด้วยการเลื่อนเวลา"""
    df_shifted_all = pd.DataFrame()
    for shift in shifts:
        df_shifted = df.copy()
        df_shifted['Quantity'] = df_shifted['Quantity'].shift(shift).bfill()  # แทนค่าหาย
        df_shifted_all = pd.concat([df_shifted_all, df_shifted])
    return df_shifted_all

def scale_features(df, scale_factors=[0.95, 1.05]):
    """ปรับสเกลข้อมูล"""
    df_scaled_all = pd.DataFrame()
    for scale in scale_factors:
        df_scaled = df.copy()
        df_scaled['Quantity'] = df_scaled['Quantity'] * scale
        df_scaled_all = pd.concat([df_scaled_all, df_scaled])
    return df_scaled_all

def add_trend(df):
    """เพิ่ม trend ให้กับข้อมูล"""
    df_trend = df.copy()
    df_trend['Quantity'] = df_trend['Quantity'] * (1 + np.linspace(0, 0.1, len(df)))
    return df_trend

def save_to_csv(df, filename):
    """บันทึกข้อมูลเป็นไฟล์ CSV"""
    df.to_csv(filename, index=False)
    print(f"✅ Data saved to {filename}")

def prepare_data(df, random_seed=42):
    """เตรียมข้อมูลสำหรับการเทรนโมเดล"""
    print("🔄 เริ่มการเตรียมข้อมูล...")
    
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    
    print("➕ กำลังเพิ่ม features...")
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_new_features(df)
    df = add_rolling_features(df)
    df = df.dropna()
    
    print("🔄 กำลังทำ Data Augmentation...")
    # สร้างข้อมูลเพิ่มเติมด้วยเทคนิคต่างๆ
    np.random.seed(random_seed)
    df_noise = add_noise(df)
    df_shifted = time_shift(df)
    df_scaled = scale_features(df)
    df_trend = add_trend(df)
    
    # รวมข้อมูลที่เพิ่มขึ้น
    df_augmented = pd.concat([df, df_noise, df_shifted, df_scaled, df_trend])
    df_augmented = df_augmented.sort_values('Date').reset_index(drop=True)
    df_augmented['Quantity'].fillna(df_augmented['Quantity'].mean(), inplace=True)
    
    # บันทึกข้อมูลที่เพิ่มขึ้น
    save_to_csv(df_augmented, 'augmented_data.csv')
    
    return df_augmented

def get_product_name(product_code):
    """ดึงชื่อสินค้าจากรหัสสินค้า"""
    if product_code == "A1001":
        return " Osida shoes"
    elif product_code == "A1002":
        return " Adda shoes"
    elif product_code == "A1004":
        return " Fashion shoes"
    elif product_code == "A1034":
        return " Court Shoes"
    elif product_code == "B1002":
        return " Long socks"
    elif product_code == "B1003":
        return " Short socks"
    elif product_code == "D1003":
        return " Mask pack"

def train_pycaret_model(df_augmented, product_code):
    """ฝึกโมเดลด้วย PyCaret"""
    print(f"🔄 กำลังเทรนโมเดล PyCaret สำหรับ {product_code}...")
    
    # ตรวจสอบว่าโมเดลถูกฝึกครั้งล่าสุดเมื่อไหร่
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'ModelPycaret')
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, f'pycaret_model_{product_code}.pkl')
    metrics_path = os.path.join(model_dir, f'model_metrics_{product_code}.csv')
    name_path = os.path.join(model_dir, f'model_name_{product_code}.csv')
    date_path = os.path.join(model_dir, f'last_trained_date_{product_code}.pkl')
    
    # ตรวจสอบว่าควรเทรนโมเดลใหม่หรือไม่
    should_train = True
    if os.path.exists(model_path) and os.path.exists(date_path):
        last_trained_date = joblib.load(date_path)
        if datetime.now() - last_trained_date < timedelta(days=30):
            print(f"⏳ ยังไม่ถึงเวลาเทรนใหม่สำหรับ {product_code}")
            # โหลดโมเดลและข้อมูลที่เกี่ยวข้อง
            model = joblib.load(model_path)
            model_metrics = pd.read_csv(metrics_path)
            model_name_df = pd.read_csv(name_path)
            model_name = model_name_df.iloc[0]['model_name']
            should_train = False
    
    if should_train:
        print(f"🛠️ กำลังสร้างโมเดลใหม่สำหรับ {product_code}...")
        
        # ตั้งค่า PyCaret
        setup(
            data=df_augmented,
            target='Quantity',
            session_id=42,
            feature_selection=True,
            normalize=True,
            remove_outliers=True,
            train_size=0.6,
            fold=5,
        )
        
        # เปรียบเทียบและเลือกโมเดลที่ดีที่สุด
        best_model = compare_models()
        final_model = finalize_model(best_model)
        
        # บันทึกข้อมูลที่ใช้ในการเทรน
        X_train = get_config('X_train')
        X_train.to_csv(os.path.join(model_dir, f'transformed_training_data_{product_code}.csv'), index=False)
        
        # ดึงข้อมูลเมตริกซ์และชื่อโมเดล
        model_metrics = pull()
        model_name = best_model.__class__.__name__
        
        # ดึงพารามิเตอร์ของโมเดลที่ดีที่สุด
        model_params = best_model.get_params()
        
        # บันทึกโมเดลและข้อมูลที่เกี่ยวข้อง
        joblib.dump(final_model, model_path)
        model_metrics.to_csv(metrics_path, index=False)
        pd.DataFrame({'model_name': [model_name]}).to_csv(name_path, index=False)
        joblib.dump(datetime.now(), date_path)
        
        print(f"✅ บันทึกโมเดลและข้อมูลที่เกี่ยวข้องเรียบร้อยแล้ว")
        
        return final_model, model_metrics, model_name, model_params
    else:
        return model, model_metrics, model_name, {}

def predict_next_sales(model, df_augmented):
    """ทำนายยอดขายในวันถัดไป"""
    # ใช้ข้อมูลล่าสุดสำหรับการทำนาย
    latest_data = df_augmented.iloc[-1:].copy()
    latest_data = latest_data.drop(columns=['Quantity'])
    
    # ทำนายยอดขาย
    predicted_sales = model.predict(latest_data)
    
    # ดึงวันที่จากข้อมูลล่าสุดและเพิ่มวัน
    predicted_date = df_augmented['Date'].iloc[-1] + pd.DateOffset(days=1)
    
    return predicted_sales[0], predicted_date

# สร้าง Flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def predict_sales_api():
    print("🔄 กำลังโหลดข้อมูล...")
    df = load_dataps()
    df_preprocessed = preprocess_dataps(df)
    
    predictions = {}
    
    for product_code in product_codes:
        print(f"🔄 กำลังเทรนโมเดลสำหรับ {product_code}...")
        
        df_product = df_preprocessed[df_preprocessed['Product_code'] == product_code]
        
        if df_product.empty:
            print(f"⚠️ ไม่มีข้อมูลสำหรับ {product_code} ข้ามไป...")
            continue
        
        # เตรียมข้อมูล
        df_augmented = prepare_data(df_product)
        
        # เทรนโมเดล
        model, model_metrics, model_name, model_params = train_pycaret_model(df_augmented, product_code)
        
        # ทำนายยอดขาย
        print(f"🔮 กำลังทำนายผลสำหรับ {product_code}...")
        next_day_prediction, predicted_date = predict_next_sales(model, df_augmented)
        productname = get_product_name(product_code)
        
        # ประเมินโมเดล
        # ลบช่องว่างและแปลงเป็นตัวพิมพ์เล็กก่อนทำการกรอง
        model_name_cleaned = model_name.lower().replace(" ", "")
        
        # แสดงข้อมูลเพื่อการดีบัก
        print(f"🔍 ค้นหาเมตริกซ์สำหรับโมเดล: {model_name}")
        print(f"🔍 ชื่อโมเดลที่มีในตาราง: {model_metrics['Model'].tolist()}")
        
        # กรองข้อมูลที่ตรงกับ model_name โดยไม่สนใจพิมพ์เล็กพิมพ์ใหญ่และช่องว่าง
        filtered_model_metrics = model_metrics[model_metrics['Model'].str.lower().str.replace(" ", "") == model_name_cleaned]
        
        # ถ้าไม่พบข้อมูล ให้ลองค้นหาแบบบางส่วนของชื่อ
        if filtered_model_metrics.empty:
            print(f"⚠️ ไม่พบข้อมูลเมตริกซ์สำหรับ {model_name} แบบตรงชื่อ ลองค้นหาแบบบางส่วน...")
            
            # ลองค้นหาด้วยส่วนแรกของชื่อโมเดล
            if 'lgbm' in model_name_cleaned:
                filtered_model_metrics = model_metrics[model_metrics['Model'].str.lower().str.contains('lgbm') | 
                                                      model_metrics['Model'].str.lower().str.contains('lightgbm')]
            else:
                # ค้นหาด้วย 4 ตัวอักษรแรกของชื่อโมเดล
                filtered_model_metrics = model_metrics[model_metrics['Model'].str.lower().str.contains(model_name_cleaned[:4])]
            
            # ถ้ายังไม่พบ ให้ใช้เมตริกซ์ของโมเดลแรกในตาราง
            if filtered_model_metrics.empty and not model_metrics.empty:
                print(f"⚠️ ไม่พบเมตริกซ์ที่ตรงกับ {model_name} ใช้เมตริกซ์ของโมเดลแรกแทน")
                filtered_model_metrics = model_metrics.iloc[:1]
        
        # ดึงเมตริกซ์ที่ต้องการ
        metrics_dict = {}
        if not filtered_model_metrics.empty:
            metrics_cols = ['MAE', 'MAPE', 'RMSE', 'R2']
            available_cols = [col for col in metrics_cols if col in filtered_model_metrics.columns]
            if available_cols:
                metrics_dict = filtered_model_metrics[available_cols].iloc[0].to_dict()
                # แปลงค่าทั้งหมดเป็น float
                metrics_dict = {k: float(v) for k, v in metrics_dict.items()}
                print(f"✅ พบเมตริกซ์สำหรับ {model_name}: {metrics_dict}")
            else:
                print(f"⚠️ ไม่พบคอลัมน์เมตริกซ์ที่ต้องการในตาราง คอลัมน์ที่มี: {filtered_model_metrics.columns.tolist()}")
        else:
            print(f"⚠️ ไม่พบข้อมูลเมตริกซ์สำหรับ {model_name} ในตารางเปรียบเทียบ")
            # แสดงชื่อโมเดลทั้งหมดที่มีในตาราง
            print(f"โมเดลที่มีในตาราง: {model_metrics['Model'].unique()}")
        
        print(f"📊 ผลการประเมินสำหรับ {product_code}:")
        for metric, value in metrics_dict.items():
            print(f"{metric}: {value:.4f}")
        
        predictions[product_code + productname] = {
            'predicted_sales': int(next_day_prediction),
            'predicted_date': predicted_date.strftime("%Y-%m-%d"),
            'model_name': model_name,
            'metrics': metrics_dict,
            'model_params': model_params
        }
    
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(host='localhost', port=8885, debug=True)