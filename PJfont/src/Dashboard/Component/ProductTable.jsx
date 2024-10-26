import React from 'react';
import { Card, CardContent, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, Typography } from '@mui/material';

const ProductTable = ({ products }) => {
    return (
        <Card style={{ flex: '1 1 calc(20% - 16px)' }}>
            <CardContent>
                <Typography variant="h6">Product List</Typography>
                <TableContainer component={Paper}>
                    <Table size="small">
                        <TableHead>
                            <TableRow>
                                <TableCell>Product</TableCell>
                                <TableCell>Price</TableCell>
                                <TableCell>Stock</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {products.map((product, index) => (
                                <TableRow key={index}>
                                    <TableCell align="center">{product.name}</TableCell>
                                    <TableCell align="center">{product.unit_price}</TableCell>
                                    <TableCell align="center">{product.stock_quantity}</TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </TableContainer>
            </CardContent>
        </Card>
    );
};

export default ProductTable;
