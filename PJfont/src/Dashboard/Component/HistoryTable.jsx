import React from 'react';
import { Card, CardContent, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, Typography } from '@mui/material';

const HistoryTable = ({ historyData }) => {
    return (
        <Card style={{ flex: '1 1 calc(100% - 16px)' }}>
            <CardContent>
                <Typography variant="h6">History Prediction</Typography>
                <TableContainer component={Paper}>
                    <Table>
                        <TableHead>
                            <TableRow>
                                <TableCell>Prediction</TableCell>
                                <TableCell>Sale</TableCell>
                                <TableCell>Difference</TableCell>
                                <TableCell>Percentage of Error</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {historyData.map((row, index) => (
                                <TableRow key={index}>
                                    <TableCell>{row.prediction}</TableCell>
                                    <TableCell>{row.sale}</TableCell>
                                    <TableCell>{row.difference}</TableCell>
                                    <TableCell>{row.percentageOfError}</TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </TableContainer>
            </CardContent>
        </Card>
    );
};

export default HistoryTable;
