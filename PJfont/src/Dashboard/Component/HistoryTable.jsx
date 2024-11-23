import React, { useState } from 'react';
import { Card, CardContent, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, Typography, Box, Button } from '@mui/material';

const HistoryTable = ({ historyData }) => {
    const [currentPage, setCurrentPage] = useState(0);
    const rowsPerPage = 5;

    // คำนวณจำนวนหน้าทั้งหมด
    const totalPages = Math.ceil(historyData.length / rowsPerPage);

    // ฟังก์ชั่นสำหรับการเปลี่ยนหน้าไปหน้าถัดไป
    const handleNextPage = () => {
        if (currentPage < totalPages - 1) {
            setCurrentPage(currentPage + 1);
        }
    };

    // ฟังก์ชั่นสำหรับการกลับไปหน้าก่อนหน้า
    const handlePrevPage = () => {
        if (currentPage > 0) {
            setCurrentPage(currentPage - 1);
        }
    };

    // ดึงข้อมูลที่จะแสดงในหน้าปัจจุบัน
    const displayedData = historyData.slice(currentPage * rowsPerPage, (currentPage + 1) * rowsPerPage);

    return (
        <Card style={{ flex: '1 1 calc(100% - 16px)' }}>
            <CardContent>
                <Typography variant="h6">History Prediction</Typography>
                <TableContainer component={Paper}>
                    <Table>
                        <TableHead>
                            <TableRow>
                                <TableCell>Date</TableCell>
                                <TableCell>Prediction</TableCell>
                                <TableCell>Sale</TableCell>
                                <TableCell>Difference</TableCell>
                                <TableCell>Percentage of Confidence</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {displayedData.map((row, index) => (
                                <TableRow key={index}>
                                    <TableCell>{row.prediction_date}</TableCell>
                                    <TableCell>{row.predicted_sales}</TableCell>
                                    <TableCell>{row.actual_sales}</TableCell>
                                    <TableCell>{row.predicted_sales - row.actual_sales}</TableCell>
                                    <TableCell>{row.error_value}</TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </TableContainer>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 2 }}>
                    <Button onClick={handlePrevPage} disabled={currentPage === 0}>
                        Previous
                    </Button>
                    <Button onClick={handleNextPage} disabled={currentPage >= totalPages - 1}>
                        Next
                    </Button>
                </Box>
            </CardContent>
        </Card>
    );
};

export default HistoryTable;
