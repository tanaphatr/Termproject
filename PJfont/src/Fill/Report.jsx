import React from 'react';
import Sidebar from '../Tool/Sidebar';
import { Typography } from '@mui/material';
import SalesReport from './Component/SalesReport'; // import SalesReport component
import SalesForm from './Component/SalesForm'; // import SalesForm component

function Report() {
  return (
    <div style={{ display: 'flex' }}>
      <Sidebar />
      <div style={{ padding: '1px', flexGrow: 1, display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <Typography variant="h4" gutterBottom sx={{ textAlign: 'left', color: 'darkblue', fontWeight: 'bold' }}>Report</Typography>
        <SalesReport />
        <SalesForm />
      </div>
    </div>
  );
}

export default Report;
