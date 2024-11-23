import React, { useContext } from 'react';
import { SalesContext } from '../Context/SalesContext'; // Import Context
import { Card, CardContent, Typography, Button, Box } from '@mui/material';

const SalesReport = () => {
    const { totalSales } = useContext(SalesContext); // ดึงยอดรวมจาก Context

    const handleSave = async () => {
        const confirmSave = window.confirm("คุณแน่ใจหรือไม่ว่าต้องการบันทึกยอดขาย?");
        if (!confirmSave) return;
        try {
            const response = await fetch(`http://localhost:8888/Daily_sales`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ totalSales }),
            });
            if (!response.ok) throw new Error("Failed to save sales data");
            alert("Sales data saved successfully!");
        } catch (error) {
            console.error("Error saving sales data:", error);
        }
    };

    return (
        <Card variant="outlined">
            <CardContent>
                <Box display="flex" justifyContent="space-between" alignItems="center">
                    <Typography variant="h5" gutterBottom>Total Sales Today</Typography>
                    <Typography variant="h6" color="text.secondary">฿{totalSales.toFixed(2)}</Typography>
                    <Button variant="outlined" color="primary" onClick={handleSave}>
                        Save
                    </Button>
                </Box>
            </CardContent>
        </Card>
    );
};

export default SalesReport;
