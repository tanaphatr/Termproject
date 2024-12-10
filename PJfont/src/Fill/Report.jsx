import React, { useState, useEffect } from 'react';
import Sidebar from '../Tool/Sidebar';
import { Typography } from '@mui/material';
import SalesForm from './Component/SalesForm';
import SalesReport from './Component/SalesReport'; // นำเข้า SalesReport

function Report() {
    const [totalSales, setTotalSales] = useState(0);  // สร้าง state สำหรับเก็บยอดขายรวม

    // ฟังก์ชันในการคำนวณยอดขายรวมจาก localStorage
    const calculateTotalSales = () => {
        const storedProducts = JSON.parse(localStorage.getItem('addedProducts')) || [];
        const totalSale = storedProducts.reduce((total, product) => total + product.total, 0);
        setTotalSales(totalSale);  // อัพเดตยอดขายรวม
    };

    // ดึงข้อมูลยอดขายรวมจาก localStorage เมื่อหน้าโหลด
    useEffect(() => {
        calculateTotalSales();  // คำนวณยอดขายรวมเมื่อหน้าโหลด

        // ตั้งเวลาให้เช็คข้อมูลทุก 1 วินาที
        const intervalId = setInterval(() => {
            calculateTotalSales();  // คำนวณยอดขายใหม่
        }, 500);

        // เคลียร์ interval เมื่อ component ถูกทำลาย
        return () => clearInterval(intervalId);
    }, []);  // useEffect จะทำงานแค่ครั้งเดียวเมื่อหน้าโหลด

    const handleTotalSaleChange = (newTotalSale) => {
        setTotalSales(newTotalSale);  // อัพเดตยอดขายรวม
    };

    return (
        <div style={{ display: 'flex' }}>
            <Sidebar />
            <div style={{ padding: '1px', flexGrow: 1, display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <Typography variant="h4" gutterBottom sx={{ textAlign: 'left', color: 'darkblue', fontWeight: 'bold' }}>
                    Report
                </Typography>
                <SalesReport totalSales={totalSales} />  {/* ส่งยอดขายรวมไปยัง SalesReport */}
                <SalesForm onTotalSaleChange={handleTotalSaleChange} />
            </div>
        </div>
    );
}

export default Report;
