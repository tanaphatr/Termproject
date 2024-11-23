import joblib
import pandas as pd
from Datafile.load_data import load_data
from Preprocess.preprocess_data import preprocess_data
from pycaret.regression import setup, compare_models, finalize_model, pull
from datetime import datetime

def train_model():
    # โหลดข้อมูลและประมวลผลข้อมูล
    df = load_data()
    df = preprocess_data(df)
    
    # เลือกฟีเจอร์ที่ต้องการใช้
    target = 'sales_amount'

    # ลบ sale_date ออกจากข้อมูล (ไม่จำเป็นต้องใส่ใน ignore_features ถ้าลบไปแล้ว)
    df = df.drop(columns=['sale_date'])
    
    # ตั้งค่าการเตรียมข้อมูลใน PyCaret
    setup(data=df, target=target, session_id=123, feature_selection=True, train_size=0.8)
    
    # เปรียบเทียบโมเดลต่างๆ ที่ PyCaret แนะนำ
    best_model = compare_models()

    # ฝึกและเลือกโมเดลที่ดีที่สุด
    final_model = finalize_model(best_model)
    
    # ดึงผลลัพธ์ metrics ที่เกี่ยวข้อง
    model_metrics = pull()  # ดึงข้อมูลผลลัพธ์จาก evaluate_model
    
    # เก็บชื่อของโมเดลที่เลือก
    model_name = best_model.__class__.__name__
    
    # บันทึกโมเดลลงในไฟล์
    model_path = 'E:/Term project/PJ/PJML/Model/model.pkl'
    joblib.dump(final_model, model_path)
    
    # บันทึก metrics ลงใน CSV
    model_metrics_path = 'E:/Term project/PJ/PJML/Model/model_metrics.csv'
    model_metrics.to_csv(model_metrics_path, index=False)
    
    # บันทึกเวลาในการฝึกโมเดล
    model_time_path = 'E:/Term project/PJ/PJML/Model/model_train_time.txt'
    with open(model_time_path, 'w') as f:
        f.write(str(datetime.now()))
    
    # คืนค่าของโมเดลที่ฝึกเสร็จแล้ว
    return final_model, model_metrics, model_name
