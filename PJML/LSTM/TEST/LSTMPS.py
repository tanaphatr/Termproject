from flask import Flask, jsonify
import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense # type: ignore

# เพิ่ม path ของโปรเจค
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'PJML')))

from Datafile.load_data import load_dataps
from Preprocess.preprocess_data import preprocess_dataps

# สร้าง Flask App
app = Flask(__name__)

# ฟังก์ชันสำหรับสร้างชุดข้อมูลลำดับเวลา (Time series)
def create_dataset(data, time_step=30):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i+time_step)])
        y.append(data[i + time_step, 0])  # ปริมาณสินค้าที่ต้องเตรียม
    return np.array(X), np.array(y)

@app.route('/', methods=['GET'])
def predict_sales():
    # เตรียมข้อมูล
    dfps = load_dataps()
    df = preprocess_dataps(dfps)  # ใช้ฟังก์ชัน preprocess_dataps ที่คุณมี
    df = df[['Product_code', 'Year', 'Month', 'Monthly_Total_Quantity']]  # เลือกคอลัมน์ที่เกี่ยวข้อง

    # ทำการสเกลข้อมูลให้เป็น 0-1
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_grouped = df.groupby('Product_code')

    # สร้างผลลัพธ์ที่ทำนายและบันทึกลง CSV
    all_predictions = []

    # สร้างโมเดลแยกตามแต่ละ Product_code
    for product_code, product_data in df_grouped:
        print(f"Training model for Product_code: {product_code}")
        
        # ทำการสเกลข้อมูลสำหรับแต่ละ Product_code
        scaled_product_data = scaler.fit_transform(product_data[['Monthly_Total_Quantity']])
        
        # สร้างชุดข้อมูลลำดับเวลา (X, y)
        X, y = create_dataset(scaled_product_data, time_step=30)
        
        # ถ้าขนาดของ X_train ไม่ถูกต้อง ให้ข้ามการฝึกโมเดลนี้
        if X.shape[0] == 0 or X.shape[1] == 0 or X.shape[2] == 0:
            print(f"Skipping Product_code {product_code} because X_train is empty.")
            continue
        
        # แบ่งข้อมูลเป็นข้อมูลฝึกและทดสอบ
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # สร้างโมเดล LSTM สำหรับแต่ละ Product_code
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=1))  # ทำนายปริมาณสินค้าที่ต้องเตรียม
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # ฝึกโมเดล
        model.fit(X_train, y_train, epochs=20, batch_size=32)
        
        # ทำนายเดือนถัดไป (จากข้อมูลล่าสุด)
        latest_data = scaled_product_data[-30:]  # ใช้ข้อมูลจาก 30 วันสุดท้าย
        latest_data = latest_data.reshape((1, latest_data.shape[0], latest_data.shape[1]))  # ปรับรูปแบบข้อมูลให้เหมาะสมกับ LSTM
        
        next_month_prediction = model.predict(latest_data)  # ทำนายเดือนถัดไป
        next_month_prediction = np.zeros((next_month_prediction.shape[0], scaled_product_data.shape[1]))
        next_month_prediction[:, 0] = next_month_prediction[:, 0]  # เติมค่าปริมาณสินค้า (Quantity) ที่ทำนาย
        next_month_prediction = scaler.inverse_transform(next_month_prediction)  # แปลงกลับเป็นข้อมูลเดิม

        # คำนวณเดือนถัดไปจากข้อมูลล่าสุด
        last_month = int(product_data['Month'].iloc[-1])  # แปลงเป็น integer
        last_year = int(product_data['Year'].iloc[-1])  # แปลงเป็น integer
        next_month = (last_month % 12) + 1  # คำนวณเดือนถัดไป
        next_year = last_year if next_month != 1 else last_year + 1  # ถ้าเดือนถัดไปเป็นมกราคม เพิ่มปี

        # เพิ่มเดือนที่ทำนายเข้าไปใน DataFrame
        prediction_month = f"Next_Month_{next_year}-{next_month:02d}"
        next_month_df = pd.DataFrame(next_month_prediction, columns=['Predicted_Monthly_Total_Quantity'])
        next_month_df['Product_code'] = product_code  # เพิ่มรหัสสินค้า
        next_month_df['Prediction_Type'] = prediction_month  # เพิ่มคอลัมน์เพื่อบ่งชี้เดือนถัดไป
        all_predictions.append(next_month_df)

    # รวมผลลัพธ์ทั้งหมด
    final_predictions = pd.concat(all_predictions, ignore_index=True)

    # บันทึกผลลัพธ์ลงในไฟล์ CSV
    final_predictions.to_csv('predictions_output.csv', index=False)

    # แปลง DataFrame เป็น JSON และส่งกลับ
    return jsonify(final_predictions.to_dict(orient='records'))



if __name__ == '__main__':
    app.run(host='localhost', port=8885, debug=True)
