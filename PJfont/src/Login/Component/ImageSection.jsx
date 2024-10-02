import React from "react";
import { Box } from "@mui/material";
import logo from "./Image/logo-color.png"; // Adjust the path according to your folder structure

const ImageSection = () => {
  return (
    <Box
      sx={{
        flex: 1,
        display: "flex",
        alignItems: "center",
        justifyContent: 'flex-start',
        height: '100%', 
      }}
    >
      <img src={logo} alt="Logo" style={{ maxWidth: "450px", height: "450px"}} /> 
    </Box>
  );
};

export default ImageSection;
