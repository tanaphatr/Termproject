import os
import sys
from pycaret.regression import *

# กำหนดเส้นทางให้เข้าถึงโมดูล
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'PJML')))

from Datafile.load_data import load_data
from Preprocess.preprocess_data import preprocess_data

# โหลดและประมวลผลข้อมูล
df = load_data()
df = preprocess_data(df)

# สร้างโมเดล PyCaret พร้อมการเลือกฟีเจอร์อัตโนมัติ
def Model(df):

    # ลบข้อมูลที่ไม่เกี่ยวข้อง
    if 'sale_date' in df.columns:
        df = df.drop(columns=['sale_date'])

    # ตั้งค่าการเตรียมข้อมูลใน PyCaret
    setup(data=df, target='sales_amount', session_id=123, 
          feature_selection=True, 
          categorical_features=df.select_dtypes(include=['object']).columns.tolist(),
          numeric_imputation='mean',
          train_size=0.8)
    
    # เปรียบเทียบโมเดล
    best_model = compare_models()
    
    # ปรับจูนโมเดลที่ดีที่สุด
    tuned_model = tune_model(best_model)
    
    # บันทึกโมเดล
    save_model(tuned_model, 'sales_prediction_with_fs_model')
    print("โมเดลถูกบันทึกเรียบร้อยแล้ว: sales_prediction_with_fs_model.pkl")
    
    return tuned_model

# เรียกใช้ฟังก์ชันสร้างโมเดล
sales_model_with_fs = Model(df)
