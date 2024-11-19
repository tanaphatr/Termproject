import sys
import os

# เพิ่ม path ของโปรเจค
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'PJML')))

import pandas as pd
from Datafile.load_data import load_data  # ฟังก์ชันสำหรับโหลดข้อมูล
from pycaret.regression import setup, compare_models, finalize_model, save_model, pull

def train_model():
    # โหลดข้อมูลจากไฟล์หรือแหล่งข้อมูล
    df = load_data()
    
    # ตรวจสอบข้อมูลเบื้องต้น
    print("ข้อมูลเริ่มต้น:")
    print(df.info())
    print(df.describe())
    
    # ลบคอลัมน์ที่ไม่จำเป็น เช่น sale_date (หากต้องการใช้ PyCaret)
    if 'sale_date' in df.columns:
        df = df.drop(columns=['sale_date'])
    
    # ตั้งค่าการเตรียมข้อมูลด้วย PyCaret
    setup_config = setup(
        data=df, 
        target='sales_amount',  # คอลัมน์เป้าหมาย
        session_id=123,  # ค่า seed สำหรับการสุ่ม
        feature_selection=True,  # เลือกฟีเจอร์ที่สำคัญ
        normalize=True,  # ทำการปรับค่าข้อมูลให้อยู่ในช่วงเดียวกัน
        remove_multicollinearity=True,  # ลบฟีเจอร์ที่มีความสัมพันธ์สูง
        multicollinearity_threshold=0.95,  # เกณฑ์ความสัมพันธ์
        categorical_imputation='mode',  # เติมค่าที่ขาดหายใน categorical ด้วย mode
        numeric_imputation='mean',  # เติมค่าที่ขาดหายใน numeric ด้วย mean
        fix_imbalance=False,  # สำหรับ classification (ปิดไว้ใน regression)
        silent=True  # ปิดการ prompt ยืนยัน
    )
    
    # เปรียบเทียบโมเดลต่าง ๆ และเลือกโมเดลที่ดีที่สุด
    print("กำลังเปรียบเทียบโมเดล...")
    best_model = compare_models()
    
    # สรุปโมเดลที่ดีที่สุด
    print(f"โมเดลที่ดีที่สุด: {best_model.__class__.__name__}")
    
    # ฝึกโมเดลขั้นสุดท้าย
    final_model = finalize_model(best_model)
    
    # ดึงข้อมูลเมตริกและผลลัพธ์การประเมิน
    model_metrics = pull()
    print("ผลลัพธ์เมตริก:")
    print(model_metrics)
    
    # บันทึกโมเดลที่ดีที่สุด
    save_model(final_model, 'best_model')
    print("โมเดลที่ดีที่สุดถูกบันทึกในไฟล์ 'best_model.pkl'")
    
    # คืนค่าผลลัพธ์
    return final_model, model_metrics, best_model.__class__.__name__

# เรียกใช้งานฟังก์ชัน
if __name__ == "__main__":
    final_model, metrics, model_name = train_model()
    print(f"ชื่อโมเดลที่เลือก: {model_name}")
    print(metrics)