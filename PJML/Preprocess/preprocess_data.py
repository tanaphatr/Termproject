import pandas as pd

def preprocess_data(df):
    # แปลงปีพุทธศักราชเป็นปีคริสต์ศักราช
    df['sale_date'] = pd.to_datetime(df['sale_date'].astype(str).str.replace(r'(\d{4})', lambda x: str(int(x.group(0)) - 543), regex=True), errors='coerce')

    # Log จำนวนวันที่ขาดหาย
    print(f"Missing sale_date entries: {df['sale_date'].isnull().sum()}")

    # แทนที่ค่าที่ขาดหายใน sales_amount และ profit_amount ด้วยค่าเฉลี่ย
    df['sales_amount'].fillna(df['sales_amount'].mean(), inplace=True)
    df['profit_amount'].fillna(df['profit_amount'].mean(), inplace=True)

    # แปลงข้อมูลหมวดหมู่เป็น One-Hot Encoding
    categorical_features = ['event', 'festival', 'weather', 'Back_to_School_Period']
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    # เพิ่มคอลัมน์ day_of_year
    df['day_of_year'] = df['sale_date'].dt.dayofyear

    # เปลี่ยนชื่อคอลัมน์ที่มีช่องว่างเป็นชื่อที่ใช้ _ แทนช่องว่าง
    df.columns = df.columns.str.replace(' ', '_')

    # สร้างไฟล์ CSV หลังจากประมวลผลข้อมูล
    df.to_csv("processed_data.csv", index=False)
    return df
