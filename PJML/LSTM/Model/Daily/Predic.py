from flask import Flask, jsonify
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from Datafile.load_data import load_data
from Preprocess.preprocess_data import preprocess_data
from LSTM import prepare_data, train_lstm_model1  # Import ฟังก์ชันที่เกี่ยวข้อง

def create_app():
    app = Flask(__name__)

    @app.route('/', methods=['GET'])
    def predict_sales_api():
        df = load_data()  # เรียกใช้ข้อมูลจาก load_data
        
        # ส่ง df ไปยังฟังก์ชัน preprocess_data
        df_preprocessed = preprocess_data(df)  # ตอนนี้ข้อมูล df จะถูก preprocess แล้ว

        # เตรียมข้อมูลให้เหมาะสมสำหรับ LSTM
        X, y, scaler, df_prepared = prepare_data(df_preprocessed)

        # แบ่งข้อมูลเป็น Train และ Test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

        # ปรับรูปร่างข้อมูลให้เหมาะสมกับ LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

        # ฝึกโมเดล
        model = train_lstm_model1(X_train, y_train)
        
        # ทำนายยอดขายในวันถัดไปโดยใช้ข้อมูล testing
        predicted_sales = model.predict(X_test)
        predicted_sales = scaler.inverse_transform(predicted_sales)

        # แปลง y_test กลับจากการสเกล
        y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

        # คำนวณ mse สำหรับ Testing Data
        mae = mean_absolute_error(predicted_sales, y_test_original)
        mape = mean_absolute_percentage_error(predicted_sales, y_test_original)
        r2 = r2_score(y_test_original, predicted_sales)

        # ตรวจสอบวันที่ล่าสุด
        predicted_date = df_prepared['sale_date'].iloc[-1] + pd.DateOffset(days=1)

        return jsonify({
            'predicted_sales': float(predicted_sales[-1][0]),
            'predicted_date': str(predicted_date),
            'model_name': "LSTM",
            'mae': mae,
            'mape': mape,
            'r2': r2
        })

    return app
