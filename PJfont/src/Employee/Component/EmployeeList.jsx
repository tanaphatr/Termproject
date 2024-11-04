import React, { useState, useEffect } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Button,
  Typography,
  Box,
  TextField,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Pagination,
  Grid,
} from "@mui/material";

const EmployeeList = () => {
  const [employees, setEmployees] = useState([]); // เก็บข้อมูลพนักงานจาก API
  const [open, setOpen] = useState(false);
  const [currentEmployee, setCurrentEmployee] = useState(null);
  const [editedFirstName, setEditedFirstName] = useState("");
  const [editedLastName, setEditedLastName] = useState("");
  const [editedPosition, setEditedPosition] = useState("");
  const [editedEmail, setEditedEmail] = useState("");
  const [editedPhone, setEditedPhone] = useState("");
  const [editedSalary, setEditedSalary] = useState("");
  const [editedNickname, setEditedNickname] = useState(""); // ฟิลด์ใหม่
  const [editedAge, setEditedAge] = useState(""); // ฟิลด์ใหม่
  const [editedAddress, setEditedAddress] = useState(""); // ฟิลด์ใหม่
  const [editedDistrict, setEditedDistrict] = useState(""); // ฟิลด์ใหม่
  const [editedProvince, setEditedProvince] = useState(""); // ฟิลด์ใหม่

  const [page, setPage] = useState(1); // หน้าเริ่มต้น
  const itemsPerPage = 5; // จำนวนพนักงานที่จะแสดงต่อหน้า

  useEffect(() => {
    const fetchEmployees = async () => {
      try {
        const response = await fetch("http://localhost:8888/Employees");
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        const data = await response.json();
        setEmployees(data); // ตั้งค่าข้อมูลที่ดึงมาจาก API
        console.log("Fetched Employees:", data);
      } catch (error) {
        console.error("Error fetching Employees:", error);
      }
    };
    fetchEmployees();
  }, []);

  const handleEdit = (id) => {
    const employee = employees.find((emp) => emp.employee_id === id);
    setCurrentEmployee(employee);
    setEditedFirstName(employee.first_name);
    setEditedLastName(employee.last_name);
    setEditedPosition(employee.position);
    setEditedEmail(employee.email);
    setEditedPhone(employee.phone);
    setEditedSalary(employee.salary);
    setEditedNickname(employee.nickname); // ตั้งค่าฟิลด์ใหม่
    setEditedAge(employee.age); // ตั้งค่าฟิลด์ใหม่
    setEditedAddress(employee.address); // ตั้งค่าฟิลด์ใหม่
    setEditedDistrict(employee.district); // ตั้งค่าฟิลด์ใหม่
    setEditedProvince(employee.province); // ตั้งค่าฟิลด์ใหม่
    setOpen(true);
  };

  const handleDelete = async (id) => {
    try {
      const response = await fetch(`http://localhost:8888/employees/${id}`, {
        method: "DELETE",
      });
      // ลบพนักงานออกจาก state
      setEmployees(employees.filter((employee) => employee.employee_id !== id));
    } catch (error) {
      console.error("Error deleting employee:", error);
    }
  };

  const handleSave = async () => {
    // สร้างอ็อบเจ็กต์สำหรับข้อมูลที่จะส่ง
    const updatedEmployee = {
      first_name: editedFirstName || currentEmployee.first_name,
      last_name: editedLastName || currentEmployee.last_name,
      position: editedPosition || currentEmployee.position,
      email: editedEmail || currentEmployee.email,
      phone: editedPhone || currentEmployee.phone,
      salary: editedSalary || currentEmployee.salary,
      nickname: editedNickname || currentEmployee.nickname, // ฟิลด์ใหม่
      age: editedAge || currentEmployee.age, // ฟิลด์ใหม่
      address: editedAddress || currentEmployee.address, // ฟิลด์ใหม่
      district: editedDistrict || currentEmployee.district, // ฟิลด์ใหม่
      province: editedProvince || currentEmployee.province, // ฟิลด์ใหม่
    };

    try {
      const response = await fetch(`http://localhost:8888/employees/${currentEmployee.employee_id}`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(updatedEmployee),
      });

      // อัปเดต state ของ employees ในกรณีที่การอัปเดตสำเร็จ
      setEmployees((prevEmployees) =>
        prevEmployees.map((employee) =>
          employee.employee_id === currentEmployee.employee_id
            ? { ...employee, ...updatedEmployee }
            : employee
        )
      );

      // ปิด dialog และรีเซ็ตค่าต่าง ๆ
      setOpen(false);
      setCurrentEmployee(null);
      setEditedFirstName("");
      setEditedLastName("");
      setEditedPosition("");
      setEditedEmail("");
      setEditedPhone("");
      setEditedSalary("");
      setEditedNickname(""); // รีเซ็ตฟิลด์ใหม่
      setEditedAge(""); // รีเซ็ตฟิลด์ใหม่
      setEditedAddress(""); // รีเซ็ตฟิลด์ใหม่
      setEditedDistrict(""); // รีเซ็ตฟิลด์ใหม่
      setEditedProvince(""); // รีเซ็ตฟิลด์ใหม่
    } catch (error) {
      console.error("Error saving employee:", error);
    }
  };

  // คำนวณข้อมูลที่จะแสดงในแต่ละหน้า
  const startIndex = (page - 1) * itemsPerPage;
  const currentEmployees = employees.slice(startIndex, startIndex + itemsPerPage);
  const totalPages = Math.ceil(employees.length / itemsPerPage);

  return (
    <Box sx={{ display: "flex", flexDirection: "column" }}>
      <Box sx={{ display: "flex", justifyContent: "space-between", mb: 2 }}>
        <Typography variant="h6">Employee List</Typography>
      </Box>
      <TableContainer component={Paper} sx={{ borderRadius: "8px", mb: 2 }}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell align="center" sx={{ fontWeight: "bold" }}>
                First Name
              </TableCell>
              <TableCell align="center" sx={{ fontWeight: "bold" }}>
                Last Name
              </TableCell>
              <TableCell align="center" sx={{ fontWeight: "bold" }}>
                Jobposition
              </TableCell>
              <TableCell align="center" sx={{ fontWeight: "bold" }}>
                Phone
              </TableCell>
              <TableCell align="center" sx={{ fontWeight: "bold" }}>
                Salary
              </TableCell>
              <TableCell align="center" sx={{ fontWeight: "bold" }}>
                Nickname
              </TableCell>
              <TableCell align="center" sx={{ fontWeight: "bold" }}>
                Age
              </TableCell>
              <TableCell align="center" sx={{ fontWeight: "bold" }}>
                Actions
              </TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {currentEmployees.map((employee) => (
              <TableRow key={employee.employee_id}>
                <TableCell align="center">{employee.first_name}</TableCell>
                <TableCell align="center">{employee.last_name}</TableCell>
                <TableCell align="center">{employee.position}</TableCell>
                <TableCell align="center">{employee.phone}</TableCell>
                <TableCell align="center">{employee.salary}</TableCell>
                <TableCell align="center">{employee.nickname}</TableCell>
                <TableCell align="center">{employee.age}</TableCell>
                <TableCell align="center">
                  <Button
                    variant="outlined"
                    sx={{ mr: 1 }}
                    onClick={() => handleEdit(employee.employee_id)}
                  >
                    EDIT
                  </Button>
                  <Button
                    variant="contained"
                    color="error"
                    onClick={() => handleDelete(employee.employee_id)}
                  >
                    DELETE
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      <Pagination
        count={totalPages}
        page={page}
        onChange={(event, value) => setPage(value)}
        variant="outlined"
        shape="rounded"
        sx={{ alignSelf: "center", mb: 2 }}
      />
      <Dialog open={open} onClose={() => setOpen(false)} maxWidth="lg" fullWidth PaperProps={{ style: { width: '90%', maxWidth: '1200px', } }}>
        <DialogTitle variant="h5" gutterBottom sx={{ textAlign: "left", color: "darkblue", fontWeight: "bold" }}>Edit Employee</DialogTitle>
        <DialogContent>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <TextField
                autoFocus
                margin="dense"
                label="First Name"
                fullWidth
                variant="outlined"
                value={editedFirstName}
                onChange={(e) => setEditedFirstName(e.target.value)}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                margin="dense"
                label="Last Name"
                fullWidth
                variant="outlined"
                value={editedLastName}
                onChange={(e) => setEditedLastName(e.target.value)}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                margin="dense"
                label="Jobposition"
                fullWidth
                variant="outlined"
                value={editedPosition}
                onChange={(e) => setEditedPosition(e.target.value)}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                margin="dense"
                label="Email"
                fullWidth
                variant="outlined"
                value={editedEmail}
                onChange={(e) => setEditedEmail(e.target.value)}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                margin="dense"
                label="Phone"
                fullWidth
                variant="outlined"
                value={editedPhone}
                onChange={(e) => setEditedPhone(e.target.value)}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                margin="dense"
                label="Salary"
                fullWidth
                variant="outlined"
                value={editedSalary}
                onChange={(e) => setEditedSalary(e.target.value)}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                margin="dense"
                label="Nickname" // ฟิลด์ใหม่
                fullWidth
                variant="outlined"
                value={editedNickname}
                onChange={(e) => setEditedNickname(e.target.value)}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                margin="dense"
                label="Age" // ฟิลด์ใหม่
                fullWidth
                variant="outlined"
                value={editedAge}
                onChange={(e) => setEditedAge(e.target.value)}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                margin="dense"
                label="Address" // ฟิลด์ใหม่
                fullWidth
                variant="outlined"
                value={editedAddress}
                onChange={(e) => setEditedAddress(e.target.value)}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                margin="dense"
                label="District" // ฟิลด์ใหม่
                fullWidth
                variant="outlined"
                value={editedDistrict}
                onChange={(e) => setEditedDistrict(e.target.value)}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                margin="dense"
                label="Province" // ฟิลด์ใหม่
                fullWidth
                variant="outlined"
                value={editedProvince}
                onChange={(e) => setEditedProvince(e.target.value)}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpen(false)}>Cancel</Button>
          <Button onClick={handleSave}>Save</Button>
        </DialogActions>
      </Dialog>

    </Box>
  );
};

export default EmployeeList;
