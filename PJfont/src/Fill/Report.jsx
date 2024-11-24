import React, { useState } from 'react';
import Sidebar from '../Tool/Sidebar';
import { Typography } from '@mui/material';
import SalesForm from './Component/SalesForm';
import SalesReport from './Component/SalesReport'; // นำเข้า SalesReport

function Report() {
    const [totalSales, setTotalSales] = useState(0);  // สร้าง state สำหรับเก็บยอดขายรวม

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
                <SalesReport totalSales={totalSales} />
                <SalesForm onTotalSaleChange={handleTotalSaleChange} />
            </div>
        </div>
    );
}

export default Report;
