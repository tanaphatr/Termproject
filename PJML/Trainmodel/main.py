# main.py ใน Trainmodel
import pandas as pd  # เพิ่มการนำเข้า pandas
from sklearn.linear_model import LinearRegression
from Datafile.load_data import load_data
from Preprocess.preprocess_data import preprocess_data

def train_model():
    # โหลดข้อมูลและประมวลผลข้อมูลเหมือนใน main function
    df = load_data()
    df = preprocess_data(df)
    
    # แปลง sale_date ให้เป็น datetime และฟีเจอร์ที่ต้องการ
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    df['year'] = df['sale_date'].dt.year
    df['month'] = df['sale_date'].dt.month
    df['day'] = df['sale_date'].dt.day
    df['day_of_year'] = df['sale_date'].dt.dayofyear

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

    X = df[features]
    y = df[target]

    # ฝึกโมเดล Linear Regression
    model = LinearRegression()
    model.fit(X, y)
    
    return model
