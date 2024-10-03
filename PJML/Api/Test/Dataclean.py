import pandas as pd

# โหลดข้อมูลจากไฟล์ CSV
df = pd.read_csv('E:/Term project/PJ/PJML/datafile/Data.csv')

# แทนที่ค่า missing เฉพาะในคอลัมน์ Sales และ Profit ด้วยค่าเฉลี่ยของแต่ละคอลัมน์
df['Sales'] = df['Sales'].fillna(round(df['Sales'].mean()))
df['Profit'] = df['Profit'].fillna(round(df['Profit'].mean()))

# บันทึก DataFrame กลับไปยังไฟล์ CSV
df.to_csv('E:/Term project/PJ/PJML/datafile/Data.csv', index=False)

print("Data has been cleaned and saved back to the CSV file.")
