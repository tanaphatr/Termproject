import React, { useEffect } from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';

const SalesReport = ({ totalSales }) => {

    return (
        <Card variant="outlined">
            <CardContent>
                <Box display="flex" justifyContent="space-between" alignItems="center">
                    <Typography variant="h5" gutterBottom>
                        Total Sales Today
                    </Typography>
                    <Typography variant="h6" color="text.secondary">
                        ฿{totalSales.toFixed(2)}  {/* แสดงยอดขาย */}
                    </Typography>
                </Box>
            </CardContent>
        </Card>
    );
};

export default SalesReport;
