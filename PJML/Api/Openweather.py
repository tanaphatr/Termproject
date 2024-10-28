import requests
from datetime import datetime, timedelta

def get_weather(lat, lon, api_key):
    # สร้าง URL สำหรับการเรียก API ข้อมูลสภาพอากาศปัจจุบัน
    url = f'http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric'
    
    try:
        # ส่งคำขอ GET ไปยัง API
        response = requests.get(url)
        response.raise_for_status()  # ตรวจสอบว่าการตอบกลับเป็นไปตามที่คาด
        data = response.json()  # แปลงการตอบกลับเป็น JSON
        return data  # ส่งคืนข้อมูลสภาพอากาศ
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")  # แสดงข้อผิดพลาดถ้ามี
        return None

def get_forecast(lat, lon, api_key):
    # สร้าง URL สำหรับการเรียก API ข้อมูลพยากรณ์อากาศ
    url = f'http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric'
    
    try:
        # ส่งคำขอ GET ไปยัง API
        response = requests.get(url)
        response.raise_for_status()  # ตรวจสอบว่าการตอบกลับเป็นไปตามที่คาด
        data = response.json()  # แปลงการตอบกลับเป็น JSON
        return data  # ส่งคืนข้อมูลพยากรณ์
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")  # แสดงข้อผิดพลาดถ้ามี
        return None

# กำหนดพิกัด latitude, longitude และ API key ของคุณ
latitude = 13.7563  # ละติจูดของกรุงเทพฯ
longitude = 100.5018  # ลองจิจูดของกรุงเทพฯ
api_key = '3dfccd2d631993bade660c38a1776d62'  # แทนที่ด้วย API key ของคุณ

# เรียกข้อมูลสภาพอากาศปัจจุบัน
current_weather = get_weather(latitude, longitude, api_key)
# เรียกข้อมูลการพยากรณ์
forecast_data = get_forecast(latitude, longitude, api_key)

# รับวันที่ปัจจุบันและวันพรุ่งนี้
today = datetime.now().date()
tomorrow = today + timedelta(days=1)

if current_weather:
    # แสดงข้อมูลสภาพอากาศปัจจุบัน
    print("Current Weather:")
    print(f"Temperature: {current_weather['main']['temp']}°C")  # อุณหภูมิปัจจุบัน
    print(f"Feels Like: {current_weather['main']['feels_like']}°C")  # รู้สึกเหมือน
    print(f"Max Temperature: {current_weather['main']['temp_max']}°C")  # อุณหภูมิสูงสุด
    print(f"Min Temperature: {current_weather['main']['temp_min']}°C")  # อุณหภูมิต่ำสุด
    print(f"Weather: {current_weather['weather'][0]['description']}")  # รายละเอียดสภาพอากาศ

if forecast_data:
    # แสดงข้อมูลพยากรณ์สำหรับวันนี้และวันพรุ่งนี้
    print("\nForecast for Today and Tomorrow:")
    daily_forecast = {}

    # ประมวลผลข้อมูลพยากรณ์
    for item in forecast_data['list']:
        date_str = item['dt_txt'].split(' ')[0]  # รับเฉพาะวันที่
        temperature = item['main']['temp']  # อุณหภูมิ
        feels_like = item['main']['feels_like']  # รู้สึกเหมือน
        weather_desc = item['weather'][0]['description']  # รายละเอียดสภาพอากาศ
        pop = item.get('pop', 0) * 100  # แปลงความน่าจะเป็นฝนเป็นเปอร์เซ็นต์

        # ตรวจสอบเฉพาะวันที่วันนี้และวันพรุ่งนี้
        if date_str == today.isoformat() or date_str == tomorrow.isoformat():
            daily_forecast[date_str] = {
                'temperature': temperature,
                'feels_like': feels_like,
                'weather_desc': weather_desc,
                'chance_of_rain': pop
            }

    # แสดงข้อมูลในแต่ละวัน
    for date, info in daily_forecast.items():
        print(f"Date: {date}")
        print(f"Temperature: {info['temperature']}°C")  # อุณหภูมิ
        print(f"Feels Like: {info['feels_like']}°C")  # รู้สึกเหมือน
        print(f"Weather: {info['weather_desc']}")  # รายละเอียดสภาพอากาศ
        print(f"Chance of Rain: {info['chance_of_rain']:.0f}%\n")  # โอกาสฝน
