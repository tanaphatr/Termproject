import React from 'react';
import Sidebar from '../Tool/Sidebar';
import { Typography } from '@mui/material';
import SalesReport from './Component/SalesReport';
import SalesForm from './Component/SalesForm';
import { SalesProvider } from './Component/SalesContext.jsx';

function Report() {
  return (
    <SalesProvider>
      <div style={{ display: 'flex' }}>
        <Sidebar />
        <div style={{ padding: '1px', flexGrow: 1, display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <Typography variant="h4" gutterBottom sx={{ textAlign: 'left', color: 'darkblue', fontWeight: 'bold' }}>
            Report
          </Typography>
          <SalesReport />
          <SalesForm />
        </div>
      </div>
    </SalesProvider>
  );
}

export default Report;
