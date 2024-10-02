import React from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

import Dashboard from './Dashboard/Dashboard'; // Adjust the import path as needed
import Home from './App'; // Example home component, adjust as necessary
import Login from './Login/Login'

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <Router>  
      <Routes>
        <Route path="/home" element={<Home />} />
        <Route path="/Dashboard" element={<Dashboard/>} />
        <Route path="/Login" element={<Login/>} />
      </Routes>
    </Router>
  </React.StrictMode>
);
