import React, { useState } from 'react';
import { Button, Box } from "@mui/material";
import ProductFormAlert from '../Component/ProductAlert';

const BTAddMore = ({ onAddProduct }) => {
  const [open, setOpen] = useState(false);

  const handleOpen = () => setOpen(true);
  const handleClose = () => setOpen(false);

  const handleSubmit = (formData) => {
    onAddProduct(formData);
    handleClose();
  };

  return (
    <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", width: "100%", maxWidth: "800px" }}>
      <Button variant="outlined" onClick={handleOpen} sx={{ mb: 2 }}>
        Add More Product
      </Button>
      <ProductFormAlert open={open} handleClose={handleClose} handleSubmit={handleSubmit} />
    </Box>
  );
};

export default BTAddMore;