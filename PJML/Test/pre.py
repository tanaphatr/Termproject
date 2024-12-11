import sys
import os
from calendar import monthrange
# เพิ่ม path ของโปรเจค
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'PJML')))

import pandas as pd
from Datafile.load_data import load_data, load_dataps

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

    # 7. รวมข้อมูลทั้งหมดกลับไปที่ DataFrame
    dfps = dfps.merge(monthly_total_quantity, on=['Year', 'Month', 'Product_code'], how='left')

    # 8. เลือกคอลัมน์ที่ต้องการแสดงผล
    dfps = dfps[['Year', 'Month', 'Product_code', 'Monthly_Total_Quantity']]

    # 9. ตัดแถวที่ Product_code ซ้ำในปีและเดือนเดียวกัน
    dfps = dfps.drop_duplicates(subset=['Year', 'Month', 'Product_code'])

    return dfps

# Main Function
if __name__ == "__main__":
    try:
        # โหลดข้อมูลจากฐานข้อมูล salesdata
        df_raw = load_data()
        print("โหลดข้อมูล salesdata สำเร็จ")

        # โหลดข้อมูลจากฐานข้อมูล product_sales
        dfps_raw = load_dataps()
        print("โหลดข้อมูล product_sales สำเร็จ")

        # เรียกใช้ฟังก์ชัน preprocess
        dfps_processed = preprocess_dataps(dfps_raw)
        print("ประมวลผลข้อมูลสำเร็จ")

        # บันทึกข้อมูลเป็นไฟล์ CSV
        output_path = os.path.join(os.path.dirname(__file__), 'preproduct_sales.csv')
        dfps_processed.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"บันทึกข้อมูลที่แปลงแล้วลงไฟล์: {output_path}")

    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")
        sys.exit(1)
