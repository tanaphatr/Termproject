import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),   '..', '..', 'PJML')))

from Datafile.load_data import load_dataps
from Preprocess.preprocess_data import preprocess_dataps

import pandas as pd

# # สมมติว่า 'preprocessed_data' เป็น DataFrame ที่คุณได้หลังจากทำการประมวลผล
# def get_top_3_products_by_day(data):
#     if isinstance(data, pd.DataFrame):
#         # กลุ่มข้อมูลตามวันที่และรหัสสินค้า
#         grouped = data.groupby(['Date', 'Product_code'])['Quantity'].sum().reset_index()

#         # จัดอันดับโดยการหายอดขายสูงสุดในแต่ละวัน
#         grouped['Rank'] = grouped.groupby('Date')['Quantity'].rank(method='first', ascending=False)

#         # เลือก 3 อันดับแรกในแต่ละวัน
#         top_3_per_day = grouped[grouped['Rank'] <= 3]

#         return top_3_per_day
#     else:
#         raise ValueError("Data is not a valid pandas DataFrame.")

# def get_overall_top_products(top_3_per_day):
#     if isinstance(top_3_per_day, pd.DataFrame):
#         # นับจำนวนการติดท็อปของแต่ละสินค้า
#         top_3_count = top_3_per_day['Product_code'].value_counts().reset_index()
#         top_3_count.columns = ['Product_code', 'Top_3_Count']

#         # จัดอันดับสินค้าโดยจำนวนการติดท็อป
#         top_3_count['Overall_Rank'] = top_3_count['Top_3_Count'].rank(method='first', ascending=False)

#         return top_3_count
#     else:
#         raise ValueError("Top 3 per day data is not a valid pandas DataFrame.")

if __name__ == "__main__":
    data = load_dataps()
    preprocessed_data = preprocess_dataps(data)

    # # หาสินค้าท็อป 3 ในแต่ละวัน
    # top_3_per_day = get_top_3_products_by_day(preprocessed_data)

    # # หาสินค้าที่ติดท็อปบ่อยที่สุด
    # overall_top_products = get_overall_top_products(top_3_per_day)

    print(preprocessed_data)
