import datetime

# ฟังก์ชันเพื่อหาวันและเดือนจากวันที่/เดือน/ปีไทย
def get_day_and_month_in_thai(day, month, year_thai):
    # แปลงปีไทยเป็นปีคริสต์ศักราช
    year_gregorian = year_thai - 543
    
    # สร้าง object วันที่
    date = datetime.date(year_gregorian, month, day)
    
    # ชื่อวันและเดือนในภาษาไทย
    days_in_thai = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]
    months_in_thai = ["มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
                      "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"]
    
    # หาวันในสัปดาห์ (0 = จันทร์, 6 = อาทิตย์)
    day_of_week = date.weekday()
    
    # คืนค่าชื่อวันและเดือน
    return f"วัน{days_in_thai[day_of_week]}ที่ {day} {months_in_thai[month-1]} {year_thai}"

# ตัวอย่างการใช้งาน
day = 15
month = 5
year_thai = 2566

print(get_day_and_month_in_thai(day, month, year_thai))
