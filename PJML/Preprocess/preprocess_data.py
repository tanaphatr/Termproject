import pandas as pd

def preprocess_data(df):
    # แปลงปีพุทธศักราชเป็นปีคริสต์ศักราช
    df['sale_date'] = pd.to_datetime(df['sale_date'].astype(str).str.replace(r'(\d{4})', lambda x: str(int(x.group(0)) - 543), regex=True), errors='coerce')

    # Log จำนวนวันที่ขาดหาย
    print(f"Missing sale_date entries: {df['sale_date'].isnull().sum()}")

    # แทนที่ค่าที่ขาดหายใน sales_amount และ profit_amount ด้วยค่าเฉลี่ย
    df['sales_amount'].fillna(df['sales_amount'].mean(), inplace=True)
    df['profit_amount'].fillna(df['profit_amount'].mean(), inplace=True)

    # แปลง event, festival, weather, Temperature, Back_to_School_Period, Seasonal เป็นรหัสตัวเลข
    df['event'], _ = pd.factorize(df['event'])
    df['day_of_week'], _ = pd.factorize(df['day_of_week'])
    df['festival'], _ = pd.factorize(df['festival'])
    df['weather'], _ = pd.factorize(df['weather'])
    df['Back_to_School_Period'], _ = pd.factorize(df['Back_to_School_Period'])
    df['Seasonal'], _ = pd.factorize(df['Seasonal'])

    # เพิ่มคอลัมน์ day_of_year
    df['day_of_year'] = df['sale_date'].dt.dayofyear

    return df
