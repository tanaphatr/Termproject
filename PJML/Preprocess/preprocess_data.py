import pandas as pd
from calendar import monthrange

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

def preprocess_dataps(dfps):
    # 1. แปลงคอลัมน์ 'Date' ให้เป็น string (ถ้าคอลัมน์ 'Date' เป็น datetime)
    dfps['Date'] = dfps['Date'].astype(str)

    # 2. เพิ่มคอลัมน์ 'Year' และ 'Month' โดยใช้ข้อมูลวันที่ที่มีอยู่
    dfps['Year'] = dfps['Date'].str[:4]  # ใช้ 4 หลักแรกของ 'Date' เพื่อดึงปี
    dfps['Month'] = dfps['Date'].str[5:7]  # ใช้หลักที่ 5 ถึง 7 เพื่อดึงเดือน
    dfps['Day'] = dfps['Date'].str[8:10]  # ดึงข้อมูลวันที่

    # 3. ตรวจสอบข้อมูลเดือนสุดท้าย
    latest_year = dfps['Year'].max()
    latest_month = dfps[dfps['Year'] == latest_year]['Month'].max()

    # ตรวจสอบจำนวนวันที่มีข้อมูลในเดือนสุดท้าย
    latest_month_data = dfps[(dfps['Year'] == latest_year) & (dfps['Month'] == latest_month)]
    days_in_month = latest_month_data['Day'].astype(int).nunique()
    total_days_in_month = monthrange(int(latest_year), int(latest_month))[1]

    # หากเดือนสุดท้ายมีข้อมูลไม่ครบเดือน ให้ลบข้อมูลเดือนนั้น
    if days_in_month < total_days_in_month:
        dfps = dfps[~((dfps['Year'] == latest_year) & (dfps['Month'] == latest_month))]

    # 4. คำนวณยอดสินค้ารวมต่อเดือนตาม Product_code
    monthly_total_quantity = dfps.groupby(['Year', 'Month', 'Product_code'])['Quantity'].sum().reset_index()
    monthly_total_quantity.rename(columns={'Quantity': 'Monthly_Total_Quantity'}, inplace=True)

    # 5. สร้าง DataFrame สำหรับ Product_code ทุกรายการในแต่ละเดือน
    all_product_codes = dfps['Product_code'].unique()
    all_combinations = pd.MultiIndex.from_product(
        [dfps['Year'].unique(), dfps['Month'].unique(), all_product_codes],
        names=['Year', 'Month', 'Product_code']
    )

    # 6. รวมข้อมูล Monthly_Total_Quantity กับทุกการจับคู่ Year, Month, Product_code
    monthly_total_quantity = monthly_total_quantity.set_index(['Year', 'Month', 'Product_code'])
    monthly_total_quantity = monthly_total_quantity.reindex(all_combinations, fill_value=0).reset_index()

    # 7. รวมข้อมูลทั้งหมดกลับไปที่ DataFrame
    dfps = dfps.merge(monthly_total_quantity, on=['Year', 'Month', 'Product_code'], how='left')

    # 8. สร้างคอลัมน์ 'Date' โดยการรวมปี, เดือน, และวัน
    dfps['Date'] = dfps['Year'] + '-' + dfps['Month'] + '-' + dfps['Day']

    # 9. เลือกคอลัมน์ที่ต้องการแสดงผล
    dfps = dfps[['Date', 'Year', 'Month', 'Product_code', 'Monthly_Total_Quantity']]

    # 10. ตัดแถวที่ Product_code ซ้ำในวันที่เดียวกัน
    dfps = dfps.drop_duplicates(subset=['Date', 'Product_code'])

    return dfps