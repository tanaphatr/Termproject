import React from "react";
import { Container, Typography } from "@mui/material";
import { ArrowBack } from "@mui/icons-material"; // นำเข้าไอคอนลูกศร
import { Link } from "react-router-dom"; // นำเข้า Link สำหรับการนำทาง
import ImageSection from "./Component/ImageSection";
import LoginForm from "./Component/LoginForm";

const Login = () => {
  const handleLogin = (loginData) => {
    // Handle login logic here
    console.log("Login attempted with:", loginData);
  };

  return (
    <div>
      <Link to="/" style={{ display: "flex", alignItems: "center", margin: "1rem" }}>
        <ArrowBack /> {/* ไอคอนลูกศร */}
        <Typography style={{ marginLeft: "0.5rem" , fontWeight: 'bold'}}>Back</Typography>
      </Link>
      <Container
        maxWidth="lg"
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          minHeight: "100%",
        }}
      >
        <div style={{ flex: 1, padding: "2rem" }}>
          <ImageSection />
        </div>
        <div style={{ flex: 1, padding: "2rem" }}>
          <LoginForm onLogin={handleLogin} />
        </div>
      </Container>
    </div>
  );
};

export default Login;
