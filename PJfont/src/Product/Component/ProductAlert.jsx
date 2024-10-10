import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  Grid,
} from '@mui/material';

const ProductFormAlert = ({ open, handleClose, handleSubmit }) => {
  const [formData, setFormData] = useState({
    date: '',
    name: '',
    price: '',
    quantity: ''
  });

  const handleChange = (event) => {
    const { name, value } = event.target;
    setFormData(prevData => ({
      ...prevData,
      [name]: value
    }));
  };

  const onSubmit = () => {
    handleSubmit(formData);
    handleClose();
  };

  return (
    <Dialog open={open} onClose={handleClose}>
      <DialogTitle>Add New Product</DialogTitle>
      <DialogContent>
        <Grid container spacing={2}>
          <Grid item xs={6}>
            <TextField
              fullWidth
              label="Date"
              name="date"
              type="date"
              value={formData.date}
              onChange={handleChange}
              InputLabelProps={{
                shrink: true,
              }}
              margin="normal"
            />
          </Grid>
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Product Name"
              name="name"
              value={formData.name}
              onChange={handleChange}
              margin="normal"
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              fullWidth
              label="Price"
              name="price"
              type="number" // รับเฉพาะตัวเลข
              value={formData.price}
              onChange={handleChange}
              margin="normal"
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              fullWidth
              label="Quantity"
              name="quantity"
              type="number" // รับเฉพาะตัวเลข
              value={formData.quantity}
              onChange={handleChange}
              margin="normal"
            />
          </Grid>
        </Grid>
      </DialogContent>
      <DialogActions>
        <Button onClick={handleClose}>Cancel</Button>
        <Button onClick={onSubmit} variant="contained" color="primary">
          Add Product
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default ProductFormAlert;
