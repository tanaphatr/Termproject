import sys
import os

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

    # 3. คำนวณยอดขายรวมต่อเดือนตาม Product_code
    monthly_total_sales = dfps.groupby(['Year', 'Month', 'Product_code'])['Total_Sale'].sum().reset_index()
    monthly_total_sales.rename(columns={'Total_Sale': 'Monthly_Total_Sales'}, inplace=True)

    # 4. คำนวณปริมาณการขายเฉลี่ยต่อวันตาม Product_code
    daily_avg_quantity = dfps.groupby(['Date', 'Product_code'])['Quantity'].mean().reset_index()
    daily_avg_quantity.rename(columns={'Quantity': 'Daily_Avg_Quantity'}, inplace=True)

    # 5. คำนวณยอดสินค้ารวมต่อเดือนตาม Product_code
    monthly_total_quantity = dfps.groupby(['Year', 'Month', 'Product_code'])['Quantity'].sum().reset_index()
    monthly_total_quantity.rename(columns={'Quantity': 'Monthly_Total_Quantity'}, inplace=True)

    # 6. รวมข้อมูลทั้งหมดกลับไปที่ DataFrame
    dfps = dfps.merge(monthly_total_sales, on=['Year', 'Month', 'Product_code'], how='left')
    dfps = dfps.merge(daily_avg_quantity, on=['Date', 'Product_code'], how='left')
    dfps = dfps.merge(monthly_total_quantity, on=['Year', 'Month', 'Product_code'], how='left')

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
