import React, { useEffect, useState } from 'react';
import { Card, CardContent, Typography, Button, Box } from '@mui/material';

const SalesReport = () => {
    const [totalSales, setTotalSales] = useState(0);
    const [productSales, setProductSales] = useState([]); // ใช้เพื่อเก็บข้อมูลการขายสินค้า

    useEffect(() => {
        const fetchSalesData = async () => {
            try {
                const response = await fetch("http://localhost:8888/Product_sales");
                if (!response.ok) {
                    throw new Error("Network response was not ok");
                }
                const data = await response.json();
                setProductSales(data);
                console.log("Fetched Product Sales:", data);
                
                // คำนวณยอดขายรวมสำหรับวันที่ปัจจุบัน
                const today = new Date().toISOString().split('T')[0]; // รับวันที่ในรูปแบบ YYYY-MM-DD
                const total = data.reduce((acc, sale) => {
                    // เช็คว่าวันที่ของการขายตรงกับวันที่ปัจจุบันหรือไม่
                    if (sale.date.split('T')[0] === today) {
                        return acc + sale.amount; // คำนวณยอดขายรวม
                    }
                    return acc; // หากไม่ตรงให้คืนค่าเดิม
                }, 0);
                setTotalSales(total);
            } catch (error) {
                console.error("Error fetching Product Sales:", error);
            }
        };

        fetchSalesData(); // เรียกใช้ฟังก์ชันดึงข้อมูล
    }, []);

    const handleSave = async () => {
        const confirmSave = window.confirm("คุณแน่ใจหรือไม่ว่าต้องการบันทึกยอดขาย?");
    
        if (!confirmSave) {
            return; // ถ้าผู้ใช้ไม่ยืนยันให้กลับ
        }
    
        try {
            const response = await fetch(`http://localhost:8888/Daily_sales`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ totalSales }), // ส่งยอดขายรวมไปในคำขอ
            });
    
            if (!response.ok) {
                throw new Error("Failed to save sales data");
            }
    
            alert("Sales data saved successfully!");
        } catch (error) {
            console.error("Error saving sales data:", error);
        }
    };
    

    return (
        <Card variant="outlined">
            <CardContent>
                <Box display="flex" justifyContent="space-between" alignItems="center">
                    <Typography variant="h5" gutterBottom>
                        Total Sales Today
                    </Typography>
                    <Typography variant="h6" color="text.secondary">
                        ฿{totalSales.toFixed(2)}
                    </Typography>
                    <Button variant="outlined" color="primary" onClick={handleSave}>
                        Save
                    </Button>
                </Box>
            </CardContent>
        </Card>
    );
};

export default SalesReport;
