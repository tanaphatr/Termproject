import React from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

import Dashboard from './Dashboard/Dashboard'; // Adjust the import path as needed
import Home from './App'; // Example home component, adjust as necessary

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <Router>  
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/Dashboard" element={<Dashboard/>} />
      </Routes>
    </Router>
  </React.StrictMode>
);
