import pandas as pd
from datetime import datetime

# อ่านไฟล์ CSV ที่มีข้อมูล
file_path = r'C:\Users\tanap\Downloads\3.csv'  # ใช้เส้นทางไฟล์ของคุณ
df = pd.read_csv(file_path)

# ฟังก์ชันเพื่อแปลงวันที่จาก พ.ศ. เป็น ค.ศ.
def convert_buddhist_to_gregorian(date_str):
    try:
        # แปลงวันที่ในรูปแบบ พ.ศ. (MM/DD/YYYY) เป็น ค.ศ.
        date_obj = datetime.strptime(date_str, '%m/%d/%Y')
        return date_obj.strftime('%Y/%m/%d')  # คืนค่าในรูปแบบ YYYY/MM/DD
    except Exception as e:
        print(f"Error converting date {date_str}: {e}")
        return None

# กรองคอลัมน์ที่ต้องการ (Date, Product_code, Quantity, Total_Sale)
df_filtered = df[['Date', 'Product_code', 'Quantity', 'Total_Sale']]

# เติมวันที่ในคอลัมน์ 'Date' ให้เป็นวันที่ในรูปแบบ YYYY/MM/DD
df_filtered['Date'] = df_filtered['Date'].apply(lambda x: convert_buddhist_to_gregorian(x) if isinstance(x, str) else x)

# เติมวันที่ที่ว่าง (NaN) ด้วยวันที่ล่าสุด
df_filtered['Date'] = df_filtered['Date'].fillna(method='ffill')

# ตรวจสอบข้อมูลหลังจากการกรอง
print(df_filtered.head())

# บันทึกข้อมูลใหม่ไปยังไฟล์ CSV
output_path = r'C:\Users\tanap\Downloads\3_filtered.csv'  # กำหนดเส้นทางที่จะบันทึกไฟล์ใหม่
df_filtered.to_csv(output_path, index=False)

print("ไฟล์ถูกบันทึกเรียบร้อยแล้วที่:", output_path)
