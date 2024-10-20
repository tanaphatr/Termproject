import pandas as pd
from Datafile.load_data import load_data
from Preprocess.preprocess_data import preprocess_data
from pycaret.regression import setup, compare_models

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
    data = df[features + [target]]

    # ใช้ PyCaret เพื่อเตรียมการฝึกโมเดล
    reg_setup = setup(data=data, target=target, verbose=False, session_id=42)

    # เปรียบเทียบโมเดลและหาผลลัพธ์ที่ดีที่สุด
    best_model = compare_models()

    print("Model comparison completed successfully.")

if __name__ == "__main__":
    main()
