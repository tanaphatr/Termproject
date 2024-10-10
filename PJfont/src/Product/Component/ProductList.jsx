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
  Typography,
  Box,
  TextField,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
} from "@mui/material";

const ProductList = () => {
  const [products, setProducts] = useState([
    { id: 1, name: "สินค้า01", price: "-", quantity: 0, date: "2024-10-10" },
    { id: 2, name: "สินค้า02", price: "-", quantity: 0, date: "2024-10-11" },
    { id: 3, name: "สินค้า03", price: "-", quantity: 0, date: "2024-10-12" },
    { id: 4, name: "สินค้า04", price: "-", quantity: 0, date: "2024-10-13" },
    { id: 5, name: "สินค้า05", price: "-", quantity: 0, date: "2024-10-14" },
    { id: 6, name: "สินค้า06", price: "-", quantity: 0, date: "2024-10-15" },
  ]);
  const [open, setOpen] = useState(false);
  const [currentProduct, setCurrentProduct] = useState(null);
  const [editedName, setEditedName] = useState("");
  const [editedPrice, setEditedPrice] = useState("");
  const [editedQuantity, setEditedQuantity] = useState(0);
  const [editedDate, setEditedDate] = useState(""); // สถานะใหม่สำหรับวันที่

  const handleEdit = (id) => {
    const product = products.find((product) => product.id === id);
    setCurrentProduct(product);
    setEditedName(product.name);
    setEditedPrice(product.price);
    setEditedQuantity(product.quantity);
    setEditedDate(product.date); // โหลดวันที่ลงในฟิลด์แก้ไข
    setOpen(true);
  };

  const handleDelete = (id) => {
    setProducts(products.filter((product) => product.id !== id));
  };

  const handleSave = () => {
    setProducts((prevProducts) =>
      prevProducts.map((product) =>
        product.id === currentProduct.id
          ? { ...product, name: editedName, price: editedPrice, quantity: editedQuantity, date: editedDate }
          : product
      )
    );
    setOpen(false);
    setCurrentProduct(null);
    setEditedName("");
    setEditedPrice("");
    setEditedQuantity(0);
    setEditedDate("");
  };

  return (
    <Box sx={{ display: "flex", flexDirection: "column" }}>
      <Typography variant="h4" gutterBottom sx={{ textAlign: "left", color: "darkblue", fontWeight: "bold" }}>
        Product
      </Typography>
      <Box sx={{ display: "flex", justifyContent: "space-between", mb: 2 }}>
        <Typography variant="h6">Product List</Typography>
      </Box>
      <TableContainer component={Paper} sx={{ borderRadius: "8px", mb: 2 }}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell align="center" sx={{ fontWeight: "bold" }}>Product</TableCell>
              <TableCell align="center" sx={{ fontWeight: "bold" }}>Price</TableCell>
              <TableCell align="center" sx={{ fontWeight: "bold" }}>Quantity</TableCell>
              <TableCell align="center" sx={{ fontWeight: "bold" }}>Date</TableCell> {/* คอลัมน์ Date ใหม่ */}
              <TableCell align="center" sx={{ fontWeight: "bold" }}>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {products.map((product) => (
              <TableRow key={product.id}>
                <TableCell align="center">{product.name}</TableCell>
                <TableCell align="center">{product.price}</TableCell>
                <TableCell align="center">{product.quantity}</TableCell>
                <TableCell align="center">{product.date}</TableCell> {/* แสดงวันที่ */}
                <TableCell align="center">
                  <Button variant="outlined" sx={{ mr: 1 }} onClick={() => handleEdit(product.id)}>
                    EDIT
                  </Button>
                  <Button variant="contained" color="error" onClick={() => handleDelete(product.id)}>
                    DELETE
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      <Dialog open={open} onClose={() => setOpen(false)}>
        <DialogTitle>Edit Product</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Product Name"
            type="text"
            fullWidth
            variant="outlined"
            value={editedName}
            onChange={(e) => setEditedName(e.target.value)}
          />
          <TextField
            margin="dense"
            label="Price"
            type="number"
            fullWidth
            variant="outlined"
            value={editedPrice}
            onChange={(e) => setEditedPrice(e.target.value)}
            inputProps={{ min: 0, pattern: "[0-9]*", inputMode: "numeric" }}
          />
          <TextField
            margin="dense"
            label="Quantity"
            type="number"
            fullWidth
            variant="outlined"
            value={editedQuantity}
            onChange={(e) => setEditedQuantity(Number(e.target.value))}
            inputProps={{ min: 0 }}
          />
          <TextField
            margin="dense"
            label="Date"
            type="date" // ฟิลด์วันที่
            fullWidth
            variant="outlined"
            value={editedDate}
            onChange={(e) => setEditedDate(e.target.value)}
            InputLabelProps={{
              shrink: true,
            }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpen(false)}>Cancel</Button>
          <Button onClick={handleSave}>Save</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ProductList;
