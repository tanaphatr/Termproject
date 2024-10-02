import React, { useState } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Button,
  TextField,
  Box,
  Typography,
} from "@mui/material";
import Sidebar from "../Tool/Sidebar"; // Import Sidebar

const Products = () => {
  const [products, setProducts] = useState([
    { id: 1, name: "สินค้า01", price: "-" },
    { id: 2, name: "สินค้า02", price: "-" },
    { id: 3, name: "สินค้า03", price: "-" },
    { id: 4, name: "สินค้า04", price: "-" },
    { id: 5, name: "สินค้า05", price: "-" },
  ]);

  const handleEdit = (id) => {
    // Implement edit logic here, possibly open a modal to edit the product details
    console.log("Edit product", id);
  };

  const handleDelete = (id) => {
    // Implement delete logic here, update the state to remove the product
    console.log("Delete product", id);
    setProducts(products.filter((product) => product.id !== id));
  };

  const handleConfirm = () => {
    console.log("Confirm changes");
    // Handle confirmation of changes, e.g., saving the data or sending to a server
  };

  const handleCancel = () => {
    console.log("Cancel changes");
    // Handle cancel logic, e.g., revert any unsaved changes
  };

  const handleAddMore = () => {
    console.log("Add more product");
    // Add new product, you might want to open a form or add a new row to the products
    const newId = products.length + 1;
    setProducts([
      ...products,
      { id: newId, name: `สินค้า0${newId}`, price: "-" },
    ]);
  };

  return (
    <Box sx={{ display: "flex", height: "100vh" }}>
      <Sidebar />
      {/* Main content */}
      <Box sx={{ flexGrow: 1, p: 3 }}>
        <Typography
          variant="h4"
          gutterBottom
          sx={{ textAlign: "left", color: "darkblue", fontWeight: "bold" }}
        >
          Product
        </Typography>
        {/* <Typography variant="h4" gutterBottom>Product</Typography> ใ*/}
        <Box sx={{ display: "flex", justifyContent: "space-between", mb: 2 }}>
          <Typography variant="h6">Product List</Typography>
        </Box>
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Product</TableCell>
                <TableCell>Price</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {products.map((product) => (
                <TableRow key={product.id}>
                  <TableCell>{product.name}</TableCell>
                  <TableCell>{product.price}</TableCell>
                  <TableCell>
                    <Button
                      variant="outlined"
                      sx={{ mr: 1 }}
                      onClick={() => handleEdit(product.id)}
                    >
                      EDIT
                    </Button>
                    <Button
                      variant="contained"
                      color="error"
                      onClick={() => handleDelete(product.id)}
                    >
                      DELETE
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
        <Box sx={{ mt: 2, display: "flex", justifyContent: "space-between" }}>
          <Box>
            <Button variant="contained" sx={{ mr: 1 }} onClick={handleConfirm}>
              Confirm
            </Button>
            <Button variant="outlined" color="error" onClick={handleCancel}>
              Cancel
            </Button>
          </Box>
          <Button variant="outlined" onClick={handleAddMore}>
            Add More Product
          </Button>
        </Box>
      </Box>
    </Box>
  );
};

export default Products;
