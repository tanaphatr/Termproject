import mysql.connector
from mysql.connector import Error

try:
    # เชื่อมต่อไปยังฐานข้อมูล
    connection = mysql.connector.connect(
        host='localhost',
        database='termproject',
        user='Admin',
        password='CE498termprojectsql',
        connection_timeout=10
    )

    if connection.is_connected():
        print("เชื่อมต่อฐานข้อมูลสำเร็จ")
        db_info = connection.get_server_info()
        print("เวอร์ชันเซิร์ฟเวอร์ MySQL:", db_info)

except Error as e:
    print("เกิดข้อผิดพลาดในการเชื่อมต่อ:", e)
