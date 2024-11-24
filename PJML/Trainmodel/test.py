import pandas as pd
from Datafile.load_data import load_data
from pycaret.regression import setup, compare_models, finalize_model, pull # type: ignore

def train_modeltest():
    # โหลดข้อมูล
    df = load_data()

    # เติมค่าขาดหายไปในคอลัมน์เป้าหมาย
    df['sales_amount'] = df['sales_amount'].fillna(df['sales_amount'].mean())

    # ลบข้อมูลที่ไม่เกี่ยวข้อง
    if 'sale_date' in df.columns:
        df = df.drop(columns=['sale_date'])

    # ตั้งค่าการเตรียมข้อมูลใน PyCaret
    setup(data=df, target='sales_amount', session_id=123, 
          feature_selection=True, 
          categorical_features=df.select_dtypes(include=['object']).columns.tolist(),
          numeric_imputation='mean',
          train_size=0.8)

    # เปรียบเทียบโมเดลต่างๆ
    best_model = compare_models()

    # ฝึกโมเดลที่ดีที่สุด
    final_model = finalize_model(best_model)

    # ดึงผลลัพธ์ metrics
    model_metrics = pull()
    model_name = best_model.__class__.__name__

    return final_model, model_metrics, model_name
