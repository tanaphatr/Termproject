import pandas as pd
import numpy as np
import joblib
import os  # เพิ่มการนำเข้า os เพื่อจัดการโฟลเดอร์
from Datafile.load_data import load_data
from Preprocess.preprocess_data import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge, LinearRegression, OrthogonalMatchingPursuit
from sklearn.ensemble import AdaBoostRegressor  # นำเข้า AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(model, X_test, y_test):
    """ประเมินโมเดลและแสดงผลเมตริกการประเมินผล"""
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # คำนวณ MAPE
    
    return score, mae, mse, rmse, mape

def main():
    try:
        # โหลดข้อมูล
        df = load_data()
        
        # ประมวลผลข้อมูล
        df = preprocess_data(df)
    except Exception as e:
        print(f"Error loading or processing data: {e}")
        return
    
    # แปลง sale_date ให้เป็นค่าที่เหมาะสมสำหรับโมเดล
    df['sale_date'] = pd.to_datetime(df['sale_date'])  # แปลงให้เป็น datetime ก่อน
    df['year'] = df['sale_date'].dt.year
    df['month'] = df['sale_date'].dt.month
    df['day'] = df['sale_date'].dt.day

    # ฟีเจอร์ที่ใช้ในการฝึกโมเดล
    features = [
        'year', 
        'month', 
        'day', 
        'day_of_year',  
        'event', 
        'day_of_week', 
        'festival', 
        'weather',
        'Back_to_School_Period',
        'Seasonal'
    ]

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

    # โมเดลที่ต้องการเปรียบเทียบ
    models = {
        'Bayesian Ridge': BayesianRidge(),
        'Linear Regression': LinearRegression(),
        'Orthogonal Matching Pursuit': OrthogonalMatchingPursuit(),
        'AdaBoost Regressor': AdaBoostRegressor()  # เพิ่ม AdaBoost Regressor
    }

    # สร้างโฟลเดอร์สำหรับเก็บโมเดล
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # เปรียบเทียบโมเดล
    for model_name, model in models.items():
        model.fit(X_train, y_train)  # ฝึกโมเดล
        score, mae, mse, rmse, mape = evaluate_model(model, X_test, y_test)  # ประเมินโมเดล
        
        # แสดงผลเมตริกการประเมินผลพร้อมคำอธิบาย
        print(f"{model_name}:")
        print(f"MAE (Mean Absolute Error): {mae:.4f} - แสดงถึงความผิดพลาดเฉลี่ยระหว่างค่าที่คาดการณ์และค่าจริง")
        print(f"RMSE (Root Mean Squared Error): {rmse:.4f} - แสดงถึงความผิดพลาดที่มีน้ำหนัก โดยให้ความสำคัญกับความผิดพลาดที่มากขึ้น")
        print(f"R² (Coefficient of Determination): {score:.4f} - แสดงถึงสัดส่วนของความแปรผันที่โมเดลสามารถอธิบายได้")
        print(f"MAPE (Mean Absolute Percentage Error): {mape:.4f} - แสดงถึงความผิดพลาดเฉลี่ยในรูปแบบเปอร์เซ็นต์\n")

        # บันทึกโมเดลที่ฝึกเสร็จแล้วในโฟลเดอร์ที่แยกต่างหาก
        model_filename = os.path.join(model_dir, f"{model_name.replace(' ', '_').lower()}_model.joblib")
        joblib.dump(model, model_filename)
        print(f"Saved {model_name} model to {model_filename}")

if __name__ == "__main__":
    main()
