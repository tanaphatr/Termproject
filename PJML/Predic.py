import joblib
import pandas as pd
from Datafile.load_data import load_data
from Preprocess.preprocess_data import preprocess_data

# โหลดข้อมูล
data = load_data()

# ประมวลผลข้อมูล
processed_data = preprocess_data(data)
print(processed_data)

# โหลดโมเดลที่ถูกบันทึกไว้
bayesian_ridge_model = joblib.load('E:/Term project/models/bayesian_ridge_model.joblib')
linear_regression_model = joblib.load('E:/Term project/models/linear_regression_model.joblib')
orthogonal_matching_pursuit_model = joblib.load('E:/Term project/models/orthogonal_matching_pursuit_model.joblib')

# Convert sale_date to datetime and extract relevant features
data['sale_date'] = pd.to_datetime(data['sale_date'])
data['year'] = data['sale_date'].dt.year
data['month'] = data['sale_date'].dt.month
data['day'] = data['sale_date'].dt.day
data['day_of_year'] = data['sale_date'].dt.dayofyear

# เลือกคอลัมน์ที่ต้องการเพื่อสร้าง latest_data
latest_data = data.iloc[-1][[
    'year', 
    'month', 
    'day', 
    'day_of_year',
    'profit_amount', 
    'event', 
    'day_of_week', 
    'festival', 
    'weather', 
    'Back_to_School_Period',
    'Seasonal'
    ]].values.reshape(1, -1)

# ทำนายยอดขายวันถัดไป
predicted_sales = orthogonal_matching_pursuit_model.predict(latest_data)

# ดึงวันที่จากข้อมูล
predicted_date = data.iloc[-1]['sale_date'] + pd.DateOffset(days=1)

# ยอดจริง
actual_sales = 9379  # แทนที่ด้วยยอดจริงของคุณ

# คำนวณค่าผิดพลาดและเปอร์เซ็นต์ความผิดพลาด
error = predicted_sales[0] - actual_sales
percentage_error = (error / actual_sales) * 100

print(f'ยอดขายที่ทำนายสำหรับวันถัดไป: {predicted_sales[0]} ในวันที่: {predicted_date.strftime("%Y-%m-%d")}')
print(f'ค่าผิดพลาด: {error}')
print(f'เปอร์เซ็นต์ความผิดพลาด: {percentage_error:.2f}%')