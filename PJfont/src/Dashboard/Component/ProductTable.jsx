import React, { useState } from 'react';
import {
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Typography,
  Button,
  Box,
} from '@mui/material';

const ProductTable = ({ products,mount}) => {
  const itemsPerPage = 6; // จำนวนสินค้าที่แสดงต่อหน้า
  const [currentPage, setCurrentPage] = useState(0); // สถานะของหน้าปัจจุบัน

  const totalPages = Math.ceil(products.length / itemsPerPage); // คำนวณจำนวนหน้า

  const displayedProducts = products.slice(
    currentPage * itemsPerPage,
    (currentPage + 1) * itemsPerPage
  ); // สินค้าที่จะแสดงตามหน้าปัจจุบัน

  const handleNextPage = () => {
    if (currentPage < totalPages - 1) {
      setCurrentPage(currentPage + 1);
    }
  };

  const handlePrevPage = () => {
    if (currentPage > 0) {
      setCurrentPage(currentPage - 1);
    }
  };
  
  return (
    <Card style={{ flex: '1 1 calc(20% - 16px)' }}>
      <CardContent>
        <Typography variant="h6">Prediction Sales(30Day)</Typography>
        <TableContainer component={Paper}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Product</TableCell>
                <TableCell align="center">Sales</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {displayedProducts.map((product, index) => (
                <TableRow key={index}>
                  <TableCell>{product.Product_code}</TableCell>
                  <TableCell align="center">{product.Prediction}</TableCell>
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

export default ProductTable;
