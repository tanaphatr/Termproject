# main.py
import pandas as pd
from Datafile.load_data import load_data
from Preprocess.preprocess_data import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def main():
    # โหลดข้อมูล
    df = load_data()

    # ประมวลผลข้อมูล
    df = preprocess_data(df)

    # กำหนดฟีเจอร์และตัวแปรเป้าหมายตามชื่อคอลัมน์ที่แท้จริง
    features = ['is_weekend', 
                'event_Normal Day',  # ปรับตามชื่อคอลัมน์ที่แท้จริง
                'festival_Buddhist Lent',
                'weather_Mostly Sunny\r',  # ปรับตามชื่อคอลัมน์ที่แท้จริง
                'profit_amount'] 
    target = 'sales_amount'

    # ตรวจสอบฟีเจอร์ที่ขาดหายไป
    missing_features = [feature for feature in features if feature not in df.columns]
    if missing_features:
        print(f"Missing features: {missing_features}")
        return

    # เตรียมข้อมูลสำหรับการฝึก
    X = df[features]
    y = df[target]

    # แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # สร้างและฝึกโมเดล
    model = LinearRegression()
    model.fit(X_train, y_train)

    # ประเมินโมเดล
    score = model.score(X_test, y_test)
    print(f"Model R^2 score: {score}")

    # คำนวณเมตริกการประเมินผล
    y_pred = model.predict(X_test)  # ทำนายค่าจากชุดทดสอบ
    mae = mean_absolute_error(y_test, y_pred)  # ค่าเฉลี่ยความผิดพลาดสัมบูรณ์
    mse = mean_squared_error(y_test, y_pred)  # ความผิดพลาดกำลังสองเฉลี่ย
    rmse = np.sqrt(mse)  # รากที่สองของความผิดพลาดกำลังสองเฉลี่ย

    # แสดงผลเมตริกการประเมินผล
    print(f"Mean Absolute Error (MAE): {mae}")  # แสดงผล MAE
    print(f"Mean Squared Error (MSE): {mse}")  # แสดงผล MSE
    print(f"Root Mean Squared Error (RMSE): {rmse}")  # แสดงผล RMSE

if __name__ == "__main__":
    main()
