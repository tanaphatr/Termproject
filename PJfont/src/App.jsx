import React from 'react'
import { Container, Typography } from "@mui/material";
import Sidebar from './Tool/Sidebar';
import ImageSection from "./Login/Component/ImageSection";

function Home() {
  return (
    <div style={{ display: 'flex' }}>
      <Sidebar/>
      <Container style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <br/>
        <br/>
        <Typography variant="h4" style={{ marginLeft: "0.5rem", fontWeight: 'bold' , color: '#003366'}}>Welcome Manager</Typography>
        <br/>
        <Typography variant="h4" style={{ marginLeft: "0.5rem", fontWeight: 'bold' , color: '#003366'}}>Lom Cha-am Shop</Typography>
        <br/>
        <ImageSection />
      </Container>
    </div>
  )
}

export default Home