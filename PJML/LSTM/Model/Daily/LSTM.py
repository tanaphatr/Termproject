import os
import sys
from unicodedata import bidirectional

from sklearn.discriminant_analysis import StandardScaler
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from flask import Flask, jsonify
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score, root_mean_squared_error
from tensorflow.keras.regularizers import l2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'PJML')))

from Datafile.load_data import load_data
from Preprocess.preprocess_data import preprocess_data

def add_time_features(df):
    df['day_of_week'] = df['sale_date'].dt.dayofweek
    df['month'] = df['sale_date'].dt.month
    df['quarter'] = df['sale_date'].dt.quarter
    df['year'] = df['sale_date'].dt.year
    df['day_of_year'] = df['sale_date'].dt.dayofyear
    
    # Add cyclical encoding for seasonal features
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    return df

def add_new_features(df):
    df['prev_day_diff'] = df['sales_amount'] - df['sales_amount'].shift(1)
    df['rolling_avg_60'] = df['sales_amount'].rolling(window=60).mean()
    return df

def add_lag_features(df, lags=[1, 2, 3, 7, 14, 30]):
    for lag in lags:
        df[f'sales_lag_{lag}'] = df['sales_amount'].shift(lag)
    return df

def add_rolling_features(df):
    windows = [7, 14, 30]
    for window in windows:
        df[f'sales_ma_{window}'] = df['sales_amount'].rolling(window=window).mean()
        df[f'sales_std_{window}'] = df['sales_amount'].rolling(window=window).std()
        df[f'sales_min_{window}'] = df['sales_amount'].rolling(window=window).min()
        df[f'sales_max_{window}'] = df['sales_amount'].rolling(window=window).max()
    return df

def augment_time_series(df, random_seed=42):
    np.random.seed(random_seed)  # กำหนด seed ให้ noise คงที่ทุกครั้ง
    augmented_data = pd.DataFrame()
    
    # Time shift augmentation
    for shift in [-3, -2, -1, 1, 2, 3]:
        shifted = df.copy()
        shifted['sales_amount'] = shifted['sales_amount'].shift(shift)
        augmented_data = pd.concat([augmented_data, shifted.dropna()])
    
    # Random noise augmentation with different intensities
    noise_scales = [0.03, 0.05, 0.07]
    for scale in noise_scales:
        noisy = df.copy()
        noise = np.random.normal(0, df['sales_amount'].std() * scale, len(df))
        noisy['sales_amount'] += noise
        augmented_data = pd.concat([augmented_data, noisy])
    
    # Scaling augmentation
    for scale in [0.9, 0.95, 1.05, 1.1]:
        scaled = df.copy()
        scaled['sales_amount'] = scaled['sales_amount'] * scale
        augmented_data = pd.concat([augmented_data, scaled])
    
    # Trend augmentation
    trend = df.copy()
    trend['sales_amount'] = trend['sales_amount'] * (1 + np.linspace(0, 0.1, len(df)))
    augmented_data = pd.concat([augmented_data, trend])
    
    return pd.concat([df, augmented_data]).sort_values('sale_date').reset_index(drop=True)

def save_to_csv(df, filename):
    df.to_csv(filename, index=False)
    print(f"✅ Data saved to {filename}")

def prepare_data(df):
    print("🔄 เริ่มการเตรียมข้อมูล...")
    
    df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
    df = df.dropna(subset=['sale_date'])
    
    print("➕ กำลังเพิ่ม features...")
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_new_features(df)
    df = add_rolling_features(df)
    df = df.dropna()
    
    print("🔄 กำลังทำ Data Augmentation...")
    df_augmented = augment_time_series(df)

    # Save augmented data
    save_to_csv(df_augmented, 'augmented_data.csv')

    # # Scale features
    # scaler = StandardScaler()
    # features = ['Temperature', 'day_of_week', 'month', 'quarter', 'year', 
    #             'day_of_year', 'month_sin', 'month_cos', 'day_of_week_sin', 
    #             'day_of_week_cos'] + \
    #           [col for col in df.columns if 'sales_lag_' in col or 
    #                                       'sales_ma_' in col or 
    #                                       'sales_std_' in col or 
    #                                       'sales_min_' in col or 
    #                                       'sales_max_' in col]
    
    # Scale features
    scaler = StandardScaler()
    features = ['Temperature', 'prev_day_diff', 'day_of_week', 'month','day_of_year','rolling_avg_60'] + \
              [col for col in df.columns if 'sales_lag_' in col or 
                                          'sales_ma_' in col or 
                                          'sales_std_' in col or 
                                          'sales_min_' in col or 
                                          'sales_max_' in col]
    
    df_augmented[features] = scaler.fit_transform(df_augmented[features])
    
    sequence_length = 60
    X, y = [], []
    
    print("🎯 กำลังสร้าง sequences...")
    for i in range(sequence_length, len(df_augmented)):
        X.append(df_augmented.iloc[i-sequence_length:i][features].values)
        y.append(df_augmented['sales_amount'].iloc[i])
    
    return np.array(X), np.array(y), df_augmented, scaler

def train_lstm_model(X_train, y_train, X_val, y_val):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'ModelLstm1')
    model_path = os.path.join(model_dir, 'lstm_model1.pkl')
    date_path = os.path.join(model_dir, 'last_trained_date1.pkl')
    
    os.makedirs(model_dir, exist_ok=True)
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("📥 โหลดโมเดลจากไฟล์ที่เก็บไว้แล้ว")
        
        if os.path.exists(date_path):
            last_trained_date = joblib.load(date_path)
        else:
            last_trained_date = datetime.min
        
        if datetime.now() - last_trained_date < timedelta(days=30):
            print("⏳ ยังไม่ถึงเวลาเทรนใหม่ (ต้องรออย่างน้อย 30 วัน)")
            return model
    
    print("🛠️ กำลังสร้างโมเดลใหม่...")
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        mode='min'
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    model = Sequential([
        Bidirectional(LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), kernel_regularizer=l2(0.001))),
        BatchNormalization(),
        Dropout(0.2),

        Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001))),
        BatchNormalization(),
        Dropout(0.2),

        Bidirectional(LSTM(64, kernel_regularizer=l2(0.001))),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=Huber(),
        metrics=['mae', 'mape']
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,  # เพิ่ม epoch
        batch_size=32,
        callbacks=[EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)]
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_history.png'))
    
    joblib.dump(model, model_path)
    joblib.dump(datetime.now(), date_path)
    print("✅ บันทึกโมเดลและวันที่เทรนล่าสุดเรียบร้อยแล้ว")
    
    return model

def predict_next_sales(model, X, df):
    last_sequence = X[-1].reshape(1, -1, X.shape[2])
    prediction = model.predict(last_sequence)[0][0]
    predicted_date = df['sale_date'].iloc[-1] + pd.DateOffset(days=1)
    return prediction, predicted_date

app = Flask(__name__)

@app.route('/', methods=['GET'])
def predict_sales_api():
    print("🔄 กำลังโหลดข้อมูล...")
    df = load_data()
    df_preprocessed = preprocess_data(df)
    
    X, y, df_prepared, scaler = prepare_data(df_preprocessed)
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.45 , shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6, shuffle=False)
    
    model = train_lstm_model(X_train, y_train, X_val, y_val)
    
    print("🔮 กำลังทำนายผล...")
    predicted_sales = model.predict(X_test)
    mae = mean_absolute_error(predicted_sales, y_test)
    mape = mean_absolute_percentage_error(predicted_sales, y_test)
    r2 = r2_score(y_test, predicted_sales)
    rmse = root_mean_squared_error(y_test, predicted_sales)
    
    next_day_prediction, predicted_date = predict_next_sales(model, X, df_prepared)
    
    print(f"📊 ผลการประเมิน:")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return jsonify({
        'predicted_sales': float(next_day_prediction),
        'predicted_date': str(predicted_date),
        'model_name': "LSTM with Augmentation",
        'metrics': {
            'mae': float(mae),
            'mape': float(mape),
            'r2': float(r2),
            'rmse': float(rmse)
        }
    })

if __name__ == '__main__':
    app.run(host='localhost', port=8885, debug=True)