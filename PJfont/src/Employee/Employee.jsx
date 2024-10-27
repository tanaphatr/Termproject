// Employees.jsx
import React from "react";
import { Typography, Box } from "@mui/material";
import Sidebar from "../Tool/Sidebar";
import EmployeeList from "../Employee/Component/EmployeeList"; // เปลี่ยนจาก ProductList เป็น EmployeeList
import EmployeeFormAlert from "./Component/EmployeeAlert"; // เปลี่ยนจาก ProductFormAlert เป็น EmployeeFormAlert
import BTAddMore from "./Component/BTAddMore"; // ปรับใช้ตามที่ต้องการ

const Employees = () => {
  return (
    <div style={{ display: 'flex' }}>
      <Sidebar />
      <div style={{ padding: '1px', flexGrow: 1, display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <Typography
          variant="h4"
          gutterBottom
          sx={{ textAlign: "left", color: "darkblue", fontWeight: "bold" }}
        >
          Employees
        </Typography>
        <EmployeeList /> {/* เปลี่ยนจาก ProductList เป็น EmployeeList */}
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2, mb: 2 }}>
          <BTAddMore /> {/* อาจจะต้องเปลี่ยนชื่อให้เข้ากับบริบทของพนักงาน */}
          <EmployeeFormAlert /> {/* เปลี่ยนจาก ProductFormAlert เป็น EmployeeFormAlert */}
        </Box>
      </div>
    </div>
  );
};

export default Employees;
