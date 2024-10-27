// Products.jsx
import React from "react";
import { Typography, Box } from "@mui/material";
import Sidebar from "../Tool/Sidebar";
import ProductList from "../Product/Component/ProductList";
import ProductFormAlert from "./Component/ProductAlert";
import BTAddMore from "./Component/BTAddmore";

const Products = () => {
  return (
    <div style={{ display: 'flex' }}>
      <Sidebar />
      <div style={{ padding: '1px', flexGrow: 1, display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <Typography
          variant="h4"
          gutterBottom
          sx={{ textAlign: "left", color: "darkblue", fontWeight: "bold" }}
        >
          Product
        </Typography>
        <ProductList />
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2, mb: 2 }}>
          <BTAddMore />
          <ProductFormAlert />
        </Box>
      </div>
    </div>
  );
};

export default Products;