import React, { useState, useEffect } from "react";
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
  const [products, setProducts] = useState([]); // เก็บข้อมูลจาก API
  const [open, setOpen] = useState(false);
  const [currentProduct, setCurrentProduct] = useState(null);
  const [editedName, setEditedName] = useState("");
  const [editedPrice, setEditedPrice] = useState("");
  const [editedQuantity, setEditedQuantity] = useState(0);

  useEffect(() => {
    const fetchProducts = async () => {
      try {
        const response = await fetch("http://localhost:8888/Products");
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        const data = await response.json();
        setProducts(data); // ตั้งค่าข้อมูลที่ดึงมาจาก API
        console.log("Fetched Products:", data);
      } catch (error) {
        console.error("Error fetching Products:", error);
      }
    };
    fetchProducts();
  }, []);

  const handleEdit = (id) => {
    const product = products.find((product) => product.product_id === id);
    setCurrentProduct(product);
    setEditedName(product.name);
    setEditedPrice(product.unit_price);
    setEditedQuantity(product.stock_quantity);
    setOpen(true);
  };

  const handleDelete = async (id) => {
    try {
      const response = await fetch(`http://localhost:8888/products/${id}`, { // ตรวจสอบว่าเส้นทางถูกต้อง
        method: 'DELETE',
      });
      // ลบผลิตภัณฑ์ออกจาก state
      setProducts(products.filter((product) => product.product_id !== id));
    } catch (error) {
      console.error("Error deleting product:", error);
    }
  };

  const handleSave = async () => {
    try {
      const response = await fetch(`http://localhost:8888/products/${currentProduct.product_id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          product_code: currentProduct.product_code, // รหัสผลิตภัณฑ์ที่ไม่เปลี่ยนแปลง
          name: editedName, // ชื่อผลิตภัณฑ์ที่แก้ไข
          stock_quantity: editedQuantity, // จำนวนที่เก็บ
          unit_price: editedPrice, // ราคาต่อหน่วย
          category: currentProduct.category, // ค่าประเภทที่ไม่เปลี่ยนแปลง
          min_stock_level: currentProduct.min_stock_level, // ค่าระดับสต็อกขั้นต่ำที่ไม่เปลี่ยนแปลง
        }),
      });

      // อัปเดต state ของ products ในกรณีที่การอัปเดตสำเร็จ
      setProducts((prevProducts) =>
        prevProducts.map((product) =>
          product.product_id === currentProduct.product_id
            ? {
              ...product,
              name: editedName,
              unit_price: editedPrice,
              stock_quantity: editedQuantity,
              // ค่าอื่น ๆ ยังคงเป็นค่าเดิม
            }
            : product
        )
      );

      // ปิด dialog และรีเซ็ตค่าต่าง ๆ
      setOpen(false);
      setCurrentProduct(null);
      setEditedName("");
      setEditedPrice("");
      setEditedQuantity(0);
    } catch (error) {
      console.error("Error saving product:", error);
    }
  };



  return (
    <Box sx={{ display: "flex", flexDirection: "column" }}>
      <Typography
        variant="h4"
        gutterBottom
        sx={{ textAlign: "left", color: "darkblue", fontWeight: "bold" }}
      >
        Product
      </Typography>
      <Box sx={{ display: "flex", justifyContent: "space-between", mb: 2 }}>
        <Typography variant="h6">Product List</Typography>
      </Box>
      <TableContainer component={Paper} sx={{ borderRadius: "8px", mb: 2 }}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell align="center" sx={{ fontWeight: "bold" }}>
                Product Code
              </TableCell>
              <TableCell align="center" sx={{ fontWeight: "bold" }}>
                Name
              </TableCell>
              <TableCell align="center" sx={{ fontWeight: "bold" }}>
                Unit Price
              </TableCell>
              <TableCell align="center" sx={{ fontWeight: "bold" }}>
                Stock Quantity
              </TableCell>
              <TableCell align="center" sx={{ fontWeight: "bold" }}>
                Actions
              </TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {products.map((product) => (
              <TableRow key={product.product_id}>
                <TableCell align="center">{product.product_code}</TableCell>
                <TableCell align="center">{product.name}</TableCell>
                <TableCell align="center">{product.unit_price}</TableCell>
                <TableCell align="center">{product.stock_quantity}</TableCell>
                <TableCell align="center">
                  <Button
                    variant="outlined"
                    sx={{ mr: 1 }}
                    onClick={() => handleEdit(product.product_id)}
                  >
                    EDIT
                  </Button>
                  <Button
                    variant="contained"
                    color="error"
                    onClick={() => handleDelete(product.product_id)}
                  >
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
            label="Unit Price"
            type="number"
            fullWidth
            variant="outlined"
            value={editedPrice}
            onChange={(e) => setEditedPrice(e.target.value)}
          />
          <TextField
            margin="dense"
            label="Stock Quantity"
            type="number"
            fullWidth
            variant="outlined"
            value={editedQuantity}
            onChange={(e) => setEditedQuantity(Number(e.target.value))}
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
