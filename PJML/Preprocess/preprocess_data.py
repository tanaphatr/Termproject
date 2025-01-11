import pandas as pd
from calendar import monthrange

import pandas as pd

def preprocess_data(df):
    # แปลงปีพุทธศักราชเป็นปีคริสต์ศักราช
    df['sale_date'] = pd.to_datetime(df['sale_date'].astype(str).str.replace(r'(\d{4})', lambda x: str(int(x.group(0)) - 543), regex=True), errors='coerce')

    # Log จำนวนวันที่ขาดหาย
    print(f"Missing sale_date entries: {df['sale_date'].isnull().sum()}")

    # แทนที่ค่าที่ขาดหายใน sales_amount และ profit_amount ด้วยค่าเฉลี่ย
    df['sales_amount'].fillna(df['sales_amount'].mean(), inplace=True)
    df['profit_amount'].fillna(df['profit_amount'].mean(), inplace=True)

    # เพิ่มคอลัมน์ day_of_year
    df['day_of_year'] = df['sale_date'].dt.dayofyear

    # แยกวันที่เป็น Date, Month, Year
    df['Date'] = df['sale_date'].dt.day
    df['Month'] = df['sale_date'].dt.month
    df['Year'] = df['sale_date'].dt.year

    # เปลี่ยนชื่อคอลัมน์ที่มีช่องว่างเป็นชื่อที่ใช้ _ แทนช่องว่าง
    df.columns = df.columns.str.replace(' ', '_')

    # ทำ One-Hot Encoding สำหรับทุกคอลัมน์ที่เป็น categorical
    categorical_cols = df.select_dtypes(include=['object']).columns  # เลือกคอลัมน์ประเภท object (string/categorical)
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)  # ทำ One-Hot Encoding

    # สร้างไฟล์ CSV หลังจากประมวลผลข้อมูล
    df.to_csv("processed_data111.csv", index=False)
    return df



def preprocess_dataps(dfps):
    # 1. แปลงคอลัมน์ 'Date' ให้เป็น string (ถ้าคอลัมน์ 'Date' เป็น datetime)
    dfps['Date'] = dfps['Date'].astype(str)

    # 2. เพิ่มคอลัมน์ 'Year' และ 'Month' โดยใช้ข้อมูลวันที่ที่มีอยู่
    dfps['Year'] = dfps['Date'].str[:4]  # ใช้ 4 หลักแรกของ 'Date' เพื่อดึงปี
    dfps['Month'] = dfps['Date'].str[5:7]  # ใช้หลักที่ 5 ถึง 7 เพื่อดึงเดือน

    # 3. ตรวจสอบข้อมูลเดือนสุดท้าย
    latest_year = dfps['Year'].max()
    latest_month = dfps[dfps['Year'] == latest_year]['Month'].max()

    # ตรวจสอบจำนวนวันที่มีข้อมูลในเดือนสุดท้าย
    latest_month_data = dfps[(dfps['Year'] == latest_year) & (dfps['Month'] == latest_month)]
    days_in_month = latest_month_data['Date'].str[8:10].astype(int).nunique()
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

    # 7. คำนวณค่าเฉลี่ยของ Monthly_Total_Quantity สำหรับแต่ละ Year และ Month
    monthly_avg_quantity = monthly_total_quantity.groupby(['Year', 'Month'])['Monthly_Total_Quantity'].mean().reset_index()
    monthly_avg_quantity.rename(columns={'Monthly_Total_Quantity': 'Monthly_Avg_Quantity'}, inplace=True)

    # 8. เติมค่า Monthly_Total_Quantity ที่หายไปด้วยค่าเฉลี่ย
    monthly_total_quantity = monthly_total_quantity.merge(monthly_avg_quantity, on=['Year', 'Month'], how='left')
    monthly_total_quantity['Monthly_Total_Quantity'] = monthly_total_quantity.apply(
    lambda row: row['Monthly_Avg_Quantity'] if row['Monthly_Total_Quantity'] == 0 else row['Monthly_Total_Quantity'], axis=1)

    # 9. รวมข้อมูลทั้งหมดกลับไปที่ DataFrame
    dfps = dfps.merge(monthly_total_quantity[['Year', 'Month', 'Product_code', 'Monthly_Total_Quantity']], 
                      on=['Year', 'Month', 'Product_code'], how='left')

    # 10. เลือกคอลัมน์ที่ต้องการแสดงผล
    dfps = dfps[['Year', 'Month', 'Product_code', 'Monthly_Total_Quantity']]

    # 11. ตัดแถวที่ Product_code ซ้ำในปีและเดือนเดียวกัน
    dfps = dfps.drop_duplicates(subset=['Year', 'Month', 'Product_code'])

    return dfps