// Products.jsx
import React from "react";
import { Typography, Box } from "@mui/material";
import Sidebar from "../Tool/Sidebar";
import ProductList from "../Product/Component/ProductList";
import ProductFormAlert from "./Component/ProductAlert";
import BTAddMore from "./Component/BTAddmore";

const Products = () => {
  return (
    <Box sx={{ display: "flex", height: "100vh" }}>
      <Sidebar />
      <Box sx={{ flexGrow: 1, p: 3, display: "flex", flexDirection: "column", overflow: "auto" }}>
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
      </Box>
    </Box>
  );
};

export default Products;