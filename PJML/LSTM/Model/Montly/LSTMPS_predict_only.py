import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from flask import Flask, jsonify
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'PJML')))

from Datafile.load_data import load_dataps
from Preprocess.preprocess_data import preprocess_dataps

# ตั้งค่าพารามิเตอร์ที่สามารถปรับได้โดยไม่ต้องเทรนใหม่
product_codes = ["A1002"]  # สามารถเพิ่มหรือลดรหัสสินค้าได้
SEQUENCE_LENGTH = 60  # ความยาวของลำดับข้อมูลที่ใช้ในการทำนาย (ปรับได้)
BATCH_SIZE = 32  # ขนาด batch ในการทำนาย (ปรับได้)
AUGMENTATION_INTENSITY = 0.03  # ความเข้มข้นของการเพิ่มข้อมูล (ปรับได้)
USE_AUGMENTATION = True  # เปิด/ปิดการใช้ data augmentation
LAG_FEATURES = [1, 2, 3, 7, 14, 30]  # ปรับ lag features ได้
ROLLING_WINDOWS = [7, 14, 30]  # ปรับ rolling windows ได้

def add_time_features(df):
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
    df['prev_day_diff'] = df['Quantity'] - df['Quantity'].shift(1)
    df['rolling_avg_60'] = df['Quantity'].rolling(window=60).mean()
    return df

def add_lag_features(df, lags=LAG_FEATURES):
    for lag in lags:
        df[f'sales_lag_{lag}'] = df['Quantity'].shift(lag)
    return df

def add_rolling_features(df, windows=ROLLING_WINDOWS):
    for window in windows:
        df[f'sales_ma_{window}'] = df['Quantity'].rolling(window=window).mean()
        df[f'sales_std_{window}'] = df['Quantity'].rolling(window=window).std()
        df[f'sales_min_{window}'] = df['Quantity'].rolling(window=window).min()
        df[f'sales_max_{window}'] = df['Quantity'].rolling(window=window).max()
    return df

def augment_time_series(df, random_seed=42, intensity=AUGMENTATION_INTENSITY):
    if not USE_AUGMENTATION:
        return df  # ถ้าปิดการใช้ augmentation ให้คืนค่าข้อมูลเดิม
        
    np.random.seed(random_seed)
    augmented_data = pd.DataFrame()
    
    # Time shift augmentation
    for shift in [-2, -1, 1, 2]:
        shifted = df.copy()
        shifted['Quantity'] = shifted['Quantity'].shift(shift).bfill()
        augmented_data = pd.concat([augmented_data, shifted.dropna()])
    
    # Noise augmentation with controlled intensity
    for _ in range(3):  # ลดจำนวนรอบลงจาก 5 เป็น 3
        scale = np.random.uniform(intensity * 0.5, intensity)
        noisy = df.copy()
        noise = np.random.normal(0, df['Quantity'].std() * scale, len(df))
        noisy['Quantity'] = np.clip(noisy['Quantity'] + noise, 0, None)
        augmented_data = pd.concat([augmented_data, noisy])

    # Scaling augmentation
    for _ in range(3):  # ลดจำนวนรอบลงจาก 5 เป็น 3
        scale = np.random.uniform(0.95, 1.05)
        scaled = df.copy()
        scaled['Quantity'] = scaled['Quantity'] * scale
        augmented_data = pd.concat([augmented_data, scaled])
    
    return pd.concat([df, augmented_data]).sort_values('Date').reset_index(drop=True)

def prepare_data(df, sequence_length=SEQUENCE_LENGTH):
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
    
    if USE_AUGMENTATION:
        print("🔄 กำลังทำ Data Augmentation...")
        df_augmented = augment_time_series(df)
    else:
        df_augmented = df

    # Scale features
    scaler = StandardScaler()
    features = ['prev_day_diff', 'day_of_week', 'month','day_of_year','rolling_avg_60'] + \
              [col for col in df.columns if 'sales_lag_' in col or 
                                          'sales_ma_' in col or 
                                          'sales_std_' in col or 
                                          'sales_min_' in col or 
                                          'sales_max_' in col]
    
    df_augmented[features] = scaler.fit_transform(df_augmented[features])
    
    X, y = [], []
    
    print(f"🎯 กำลังสร้าง sequences ด้วยความยาว {sequence_length}...")
    for i in range(sequence_length, len(df_augmented)):
        X.append(df_augmented.iloc[i-sequence_length:i][features].values)
        y.append(df_augmented['Quantity'].iloc[i])
    
    return np.array(X), np.array(y), df_augmented, scaler

def load_trained_model(product_code):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'ModelLstm2')
    model_path = os.path.join(model_dir, f'lstm_model_{product_code}.pkl')
    best_params_path = os.path.join(model_dir, f'best_params_{product_code}.pkl')
    
    if not os.path.exists(model_path):
        print(f"❌ ไม่พบโมเดลสำหรับ {product_code} ที่ {model_path}")
        return None, None
    
    model = joblib.load(model_path)
    print(f"📥 โหลดโมเดล {model_path} สำเร็จ")
    
    if os.path.exists(best_params_path):
        best_params = joblib.load(best_params_path)
        print(f"📥 โหลดพารามิเตอร์ที่ดีที่สุด {best_params_path} สำเร็จ")
        print(f"🔍 พารามิเตอร์ที่ดีที่สุด: {best_params}")
    else:
        best_params = None
        print(f"⚠️ ไม่พบไฟล์พารามิเตอร์ที่ดีที่สุดสำหรับ {product_code}")
    
    return model, best_params

def predict_next_sales(model, X, df):
    # ปรับขนาด batch ในการทำนาย
    last_sequence = X[-1].reshape(1, -1, X.shape[2])
    prediction = model.predict(last_sequence, batch_size=BATCH_SIZE, verbose=0)[0][0]
    predicted_date = df['Date'].iloc[-1] + pd.DateOffset(days=1)
    return prediction, predicted_date

def get_product_name(product_code):
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

def evaluate_model(model, X_test, y_test):
    # ปรับขนาด batch ในการประเมินผล
    predicted_sales = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    mae = mean_absolute_error(predicted_sales, y_test)
    mape = mean_absolute_percentage_error(predicted_sales, y_test)
    rmse = root_mean_squared_error(predicted_sales, y_test)
    r2 = r2_score(y_test, predicted_sales)
    
    return {
        'mae': round(float(mae), 2),
        'mape': round(float(mape), 2),
        'r2': round(float(r2), 2),
        'rmse': round(float(rmse), 2)
    }

def try_different_sequence_lengths(df_product, product_code, model):
    """ทดลองใช้ความยาวลำดับข้อมูลที่แตกต่างกันและดูผลลัพธ์"""
    results = {}
    
    for seq_length in [30, 45, 60, 90]:
        print(f"\n🔄 ทดสอบความยาวลำดับข้อมูล {seq_length} สำหรับ {product_code}...")
        X, y, df_prepared, _ = prepare_data(df_product, sequence_length=seq_length)
        
        # แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # ประเมินผลโมเดล
        metrics = evaluate_model(model, X_test, y_test)
        results[seq_length] = metrics
        print(f"📊 ผลการประเมินสำหรับความยาวลำดับข้อมูล {seq_length}:")
        print(f"MAE: {metrics['mae']}, MAPE: {metrics['mape']}%, R²: {metrics['r2']}, RMSE: {metrics['rmse']}")
    
    # หาความยาวลำดับข้อมูลที่ดีที่สุด (ใช้ MAE เป็นเกณฑ์)
    best_seq_length = min(results.keys(), key=lambda k: results[k]['mae'])
    print(f"\n🏆 ความยาวลำดับข้อมูลที่ดีที่สุดสำหรับ {product_code}: {best_seq_length}")
    print(f"📊 ผลการประเมิน: MAE: {results[best_seq_length]['mae']}, MAPE: {results[best_seq_length]['mape']}%")
    
    return best_seq_length, results

def try_different_batch_sizes(X_test, y_test, model, product_code):
    """ทดลองใช้ขนาด batch ที่แตกต่างกันและดูผลลัพธ์"""
    results = {}
    
    for batch_size in [16, 32, 64, 128]:
        print(f"\n🔄 ทดสอบขนาด batch {batch_size} สำหรับ {product_code}...")
        
        # ทำนายด้วยขนาด batch ที่แตกต่างกัน
        start_time = datetime.now()
        predicted_sales = model.predict(X_test, batch_size=batch_size, verbose=0)
        prediction_time = (datetime.now() - start_time).total_seconds()
        
        # คำนวณเมทริกซ์
        mae = mean_absolute_error(predicted_sales, y_test)
        mape = mean_absolute_percentage_error(predicted_sales, y_test)
        
        results[batch_size] = {
            'mae': round(float(mae), 2),
            'mape': round(float(mape), 2),
            'prediction_time': round(prediction_time, 2)
        }
        print(f"📊 ผลการประเมินสำหรับขนาด batch {batch_size}:")
        print(f"MAE: {results[batch_size]['mae']}, MAPE: {results[batch_size]['mape']}%, เวลาทำนาย: {results[batch_size]['prediction_time']} วินาที")
    
    # หาขนาด batch ที่ดีที่สุด (ใช้ MAE เป็นเกณฑ์)
    best_batch_size = min(results.keys(), key=lambda k: results[k]['mae'])
    print(f"\n🏆 ขนาด batch ที่ดีที่สุดสำหรับ {product_code}: {best_batch_size}")
    print(f"📊 ผลการประเมิน: MAE: {results[best_batch_size]['mae']}, MAPE: {results[best_batch_size]['mape']}%")
    
    return best_batch_size, results

def try_different_augmentation_settings(df_product, product_code, model):
    """ทดลองใช้การตั้งค่า augmentation ที่แตกต่างกันและดูผลลัพธ์"""
    global USE_AUGMENTATION, AUGMENTATION_INTENSITY
    results = {}
    
    # ทดสอบการเปิด/ปิด augmentation
    for use_aug in [True, False]:
        # ทดสอบความเข้มข้นของ augmentation ถ้าเปิดใช้งาน
        intensities = [0.01, 0.03, 0.05] if use_aug else [0]
        
        for intensity in intensities:
            setting_name = f"{'Aug_ON' if use_aug else 'Aug_OFF'}_Int_{intensity}"
            print(f"\n🔄 ทดสอบการตั้งค่า augmentation: {setting_name} สำหรับ {product_code}...")
            
            # ตั้งค่าตัวแปรสุหรับการทดสอบนี้
            original_aug = USE_AUGMENTATION
            original_intensity = AUGMENTATION_INTENSITY
            USE_AUGMENTATION = use_aug
            AUGMENTATION_INTENSITY = intensity
            
            # เตรียมข้อมูลด้วยการตั้งค่าใหม่
            X, y, df_prepared, _ = prepare_data(df_product)
            
            # แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            # ประเมินผลโมเดล
            metrics = evaluate_model(model, X_test, y_test)
            results[setting_name] = metrics
            print(f"📊 ผลการประเมินสำหรับการตั้งค่า {setting_name}:")
            print(f"MAE: {metrics['mae']}, MAPE: {metrics['mape']}%, R²: {metrics['r2']}, RMSE: {metrics['rmse']}")
            
            # คืนค่าตัวแปรกลับเป็นค่าเดิม
            USE_AUGMENTATION = original_aug
            AUGMENTATION_INTENSITY = original_intensity
    
    # หาการตั้งค่าที่ดีที่สุด (ใช้ MAE เป็นเกณฑ์)
    best_setting = min(results.keys(), key=lambda k: results[k]['mae'])
    print(f"\n🏆 การตั้งค่า augmentation ที่ดีที่สุดสำหรับ {product_code}: {best_setting}")
    print(f"📊 ผลการประเมิน: MAE: {results[best_setting]['mae']}, MAPE: {results[best_setting]['mape']}%")
    
    return best_setting, results

app = Flask(__name__)
@app.route('/', methods=['GET'])
def predict_sales_api():
    print("🔄 กำลังโหลดข้อมูล...")
    df = load_dataps()
    df_preprocessed = preprocess_dataps(df)

    predictions = {}

    for product_code in product_codes:
        print(f"🔄 กำลังโหลดโมเดลสำหรับ {product_code}...")

        df_product = df_preprocessed[df_preprocessed['Product_code'] == product_code]

        if df_product.empty:
            print(f"⚠️ ไม่มีข้อมูลสำหรับ {product_code} ข้ามไป...")
            continue

        # โหลดโมเดลที่เทรนไว้แล้ว
        model, best_params = load_trained_model(product_code)
        if model is None:
            print(f"⚠️ ไม่สามารถโหลดโมเดลสำหรับ {product_code} ข้ามไป...")
            continue

        # เตรียมข้อมูลด้วยพารามิเตอร์ปัจจุบัน
        X, y, df_prepared, scaler = prepare_data(df_product)

        # แบ่งข้อมูลเพื่อทดสอบ
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, shuffle=False)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6, shuffle=False)

        print(f"🔮 กำลังทำนายผลสำหรับ {product_code}...")
        metrics = evaluate_model(model, X_test, y_test)

        next_day_prediction, predicted_date = predict_next_sales(model, X, df_prepared)
        productname = get_product_name(product_code) 

        print(f"📊 ผลการประเมินสำหรับ {product_code}:")
        print(f"MAE: {metrics['mae']}, MAPE: {metrics['mape']}%, R²: {metrics['r2']}, RMSE: {metrics['rmse']}")

        predictions[product_code + productname] = {
            'predicted_sales': int(next_day_prediction),
            'predicted_date': str(predicted_date),
            'metrics': metrics
        }
    return jsonify(predictions)

@app.route('/optimize', methods=['GET'])
def optimize_parameters_api():
    """API สำหรับหาพารามิเตอร์ที่เหมาะสมที่สุดโดยไม่ต้องเทรนใหม่"""
    print("🔄 กำลังโหลดข้อมูล...")
    df = load_dataps()
    df_preprocessed = preprocess_dataps(df)

    results = {}

    for product_code in product_codes:
        print(f"\n🔍 กำลังหาพารามิเตอร์ที่เหมาะสมสำหรับ {product_code}...")

        df_product = df_preprocessed[df_preprocessed['Product_code'] == product_code]

        if df_product.empty:
            print(f"⚠️ ไม่มีข้อมูลสำหรับ {product_code} ข้ามไป...")
            continue

        # โหลดโมเดลที่เทรนไว้แล้ว
        model, best_params = load_trained_model(product_code)
        if model is None:
            print(f"⚠️ ไม่สามารถโหลดโมเดลสำหรับ {product_code} ข้ามไป...")
            continue

        product_results = {}

        # 1. ทดสอบความยาวลำดับข้อมูลที่แตกต่างกัน
        print("\n🔍 กำลังทดสอบความยาวลำดับข้อมูลที่แตกต่างกัน...")
        best_seq_length, seq_results = try_different_sequence_lengths(df_product, product_code, model)
        product_results['best_sequence_length'] = best_seq_length
        product_results['sequence_length_results'] = {str(k): v for k, v in seq_results.items()}

        # 2. ทดสอบขนาด batch ที่แตกต่างกัน
        print("\n🔍 กำลังทดสอบขนาด batch ที่แตกต่างกัน...")
        # ใช้ความยาวลำดับข้อมูลที่ดีที่สุดที่พบ
        global SEQUENCE_LENGTH
        original_seq_length = SEQUENCE_LENGTH
        SEQUENCE_LENGTH = best_seq_length
        
        X, y, df_prepared, _ = prepare_data(df_product)
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        best_batch_size, batch_results = try_different_batch_sizes(X_test, y_test, model, product_code)
        product_results['best_batch_size'] = best_batch_size
        product_results['batch_size_results'] = {str(k): v for k, v in batch_results.items()}
        
        # 3. ทดสอบการตั้งค่า augmentation ที่แตกต่างกัน
        print("\n🔍 กำลังทดสอบการตั้งค่า augmentation ที่แตกต่างกัน...")
        best_aug_setting, aug_results = try_different_augmentation_settings(df_product, product_code, model)
        product_results['best_augmentation_setting'] = best_aug_setting
        product_results['augmentation_results'] = aug_results
        
        # คืนค่าตัวแปรกลับเป็นค่าเดิม
        SEQUENCE_LENGTH = original_seq_length
        
        # สรุปผลลัพธ์ที่ดีที่สุด
        print(f"\n📋 สรุปพารามิเตอร์ที่เหมาะสมที่สุดสำหรับ {product_code}:")
        print(f"🔹 ความยาวลำดับข้อมูลที่ดีที่สุด: {best_seq_length}")
        print(f"🔹 ขนาด batch ที่ดีที่สุด: {best_batch_size}")
        print(f"🔹 การตั้งค่า augmentation ที่ดีที่สุด: {best_aug_setting}")
        
        results[product_code] = product_results

    return jsonify(results)

if __name__ == '__main__':
    app.run(host='localhost', port=8886, debug=True)