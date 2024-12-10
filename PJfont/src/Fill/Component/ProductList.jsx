import React from 'react';
import {
    Button,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Paper,
} from '@mui/material';

const ProductList = ({ addedProducts, handleRemoveProduct }) => {
    return (
        <TableContainer component={Paper} style={{ marginTop: '20px' }}>
            <Table>
                <TableHead>
                    <TableRow>
                        <TableCell align="center">Nameproduct</TableCell>
                        <TableCell align="center">Quantity</TableCell>
                        <TableCell align="center">Unitprice</TableCell>
                        <TableCell align="center">Total</TableCell>
                        <TableCell align="center">Action</TableCell>
                    </TableRow>
                </TableHead>
                <TableBody>
                    {addedProducts.map((product) => (
                        <TableRow key={product.product_id}>
                            <TableCell align="center" style={{ verticalAlign: 'middle' }}>{product.name}</TableCell>
                            <TableCell align="center" style={{ verticalAlign: 'middle' }}>{product.quantity}</TableCell>
                            <TableCell align="center" style={{ verticalAlign: 'middle' }}>{product.unit_price}</TableCell>
                            <TableCell align="center" style={{ verticalAlign: 'middle' }}>{product.total}</TableCell>
                            <TableCell align="center" style={{ verticalAlign: 'middle' }}>
                                <Button
                                    variant="outlined"
                                    color="error"
                                    onClick={() => handleRemoveProduct(product.product_id)} // ส่ง product_id เพื่อลบ
                                >
                                    Delete
                                </Button>
                            </TableCell>
                        </TableRow>
                    ))}
                </TableBody>
            </Table>
        </TableContainer>
    );
};

export default ProductList;
