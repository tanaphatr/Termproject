import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, jsonify
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense  # type: ignore
from sklearn.model_selection import train_test_split

# เพิ่ม path ของโปรเจค
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'PJML')))

from Datafile.load_data import load_dataps
from Preprocess.preprocess_data import preprocess_dataps
from Datafile.load_data import load_data
#===============================================เตรียมข้อมูล=========================================
def prepare_data(df):
    # เติมค่าขาดหายไปในคอลัมน์เป้าหมาย
    df['sales_amount'] = df['sales_amount'].fillna(df['sales_amount'].mean())

    # แปลงวันที่จาก พ.ศ. เป็น ค.ศ.
    if 'sale_date' in df.columns:
        df['sale_date'] = df['sale_date'].astype(str)  # แปลงเป็น string
        df['sale_date'] = df['sale_date'].apply(lambda x: str(int(x[:4]) - 543) + x[4:] if len(x) == 10 else x)

        # แปลงวันที่ให้เป็น datetime และเรียงลำดับ
        df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
        df = df.sort_values('sale_date')

    # สเกลข้อมูลให้อยู่ในช่วง [0, 1]
    scaler = MinMaxScaler()
    df['sales_amount_scaled'] = scaler.fit_transform(df[['sales_amount']])

    # เตรียมข้อมูลสำหรับ LSTM
    sequence_length = 10  # จำนวนวันใน sequence
    X, y = [], []
    for i in range(sequence_length, len(df)):
        X.append(df['sales_amount_scaled'].iloc[i-sequence_length:i].values)
        y.append(df['sales_amount_scaled'].iloc[i])

    # บันทึกคอลัมน์ sale_date เป็นไฟล์ CSV
    df[['sale_date']].to_csv('sale_dates.csv', index=False)

    return np.array(X), np.array(y), scaler, df
#===============================================เทรน=========================================
def train_lstm_model(X, y):
    # สร้างโมเดล LSTM
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=32, verbose=1)
    return model
#===============================================ทำนาย=========================================
def predict_next_sales(model, X, scaler, df):
    # ใช้ข้อมูลล่าสุดในการทำนาย
    last_sequence = X[-1].reshape(1, -1, 1)
    prediction_scaled = model.predict(last_sequence)
    predicted_sales = scaler.inverse_transform(prediction_scaled)[0][0]
    
    # ดึงวันที่ล่าสุดจากคอลัมน์ 'sale_date' และเพิ่มวัน
    predicted_date = df['sale_date'].iloc[-1] + pd.DateOffset(days=1)
    return predicted_sales, predicted_date

#===============================================Time series=========================================
# ฟังก์ชันสำหรับสร้างชุดข้อมูลลำดับเวลา (Time series)
def create_dataset(data, time_step=30):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i+time_step)])
        y.append(data[i + time_step, 0])  # ปริมาณสินค้าที่ต้องเตรียม
    return np.array(X), np.array(y)

#===============================================API=========================================
# Flask App
app = Flask(__name__)

@app.route('/', methods=['GET'])
def predict_sales():
    # โหลดข้อมูล
    df = load_data()

    # เตรียมข้อมูล
    X, y, scaler, df_prepared = prepare_data(df)

    # ปรับรูปร่างข้อมูลให้เหมาะสมกับ LSTM
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # แบ่งข้อมูลเป็น training และ testing set (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # ฝึกโมเดล
    model = train_lstm_model(X_train, y_train)

    # ทำนายยอดขายในวันถัดไปโดยใช้ข้อมูล testing
    predicted_sales = model.predict(X_test)
    predicted_sales = scaler.inverse_transform(predicted_sales)

    # แปลงค่า y_test กลับจากการสเกลเป็นยอดขายจริง
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # สร้างกราฟเปรียบเทียบผลลัพธ์
    plt.figure(figsize=(10, 6))
    plt.plot(df_prepared['sale_date'].iloc[-len(y_test):], y_test_actual, label="Actual Sales", color='blue')
    plt.plot(df_prepared['sale_date'].iloc[-len(y_test):], predicted_sales, label="Predicted Sales", color='red')
    plt.xlabel('Date')
    plt.ylabel('Sales Amount')
    plt.title('Actual vs Predicted Sales')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # เตรียมข้อมูลเพิ่มเติม
    dfps = load_dataps()
    df = preprocess_dataps(dfps)
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
        next_month_df = pd.DataFrame(next_month_prediction, columns=['Predicted_Quantity'])
        next_month_df['Product_code'] = product_code  # เพิ่มรหัสสินค้า
        next_month_df['Type'] = prediction_month  # เพิ่มคอลัมน์เพื่อบ่งชี้เดือนถัดไป
        all_predictions.append(next_month_df)

    # รวมผลลัพธ์ทั้งหมด
    final_predictions = pd.concat(all_predictions, ignore_index=True)

    # บันทึกผลลัพธ์ลงในไฟล์ CSV
    final_predictions.to_csv('predictions_output.csv', index=False)

    # รวมข้อมูลเพื่อส่งกลับ
    response_data = {
        'predicted_sales': float(predicted_sales[-1][0]),
        'predicted_date': str(df_prepared['sale_date'].iloc[-1] + pd.DateOffset(days=1)),
        'model_name': "LSTM",
        'predictions': final_predictions.to_dict(orient='records')
    }

    return jsonify(response_data)
    
if __name__ == '__main__':
    app.run(host='localhost', port=8885, debug=True)
