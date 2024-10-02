import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

import Dashboard from "./Dashboard/Dashboard";
import Home from "./App";
import Login from "./Login/Login";
import Products from "./Product/Products";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <Router>
      <Routes>
        <Route path="/home" element={<Home />} />
        <Route path="/Dashboard" element={<Dashboard />} />
        <Route path="/Login" element={<Login />} />
        <Route path="/Products" element={<Products />} />
      </Routes>
    </Router>
  </React.StrictMode>
);
