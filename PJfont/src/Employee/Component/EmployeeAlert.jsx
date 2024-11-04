import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  Grid,
} from '@mui/material';
import axios from 'axios';

const EmployeeFormAlert = ({ open, handleClose }) => {
  const [formData, setFormData] = useState({
    first_name: '',
    last_name: '',
    position: '',
    email: '',
    phone: '',
    nickname: '',
    age: '',
    address: '',
    district: '',
    province: '',
    salary: ''
  });

  const handleChange = (event) => {
    const { name, value } = event.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: value,
    }));
  };

  const handleSubmit = async (data) => {
    try {
      await axios.post('http://localhost:8888/Employees', data);
      window.location.reload(); // Refresh the page
    } catch (error) {
      console.error("Error adding employee:", error);
      alert("Failed to add employee. Please try again.");
    }
  };

  const onSubmit = () => {
    handleSubmit(formData);
    handleClose();
  };

  return (
    <Dialog open={open}onClose={handleClose} maxWidth="lg"fullWidth PaperProps={{style: {width: '90%',maxWidth: '1200px',},}}>
      <DialogTitle variant="h5" gutterBottom sx={{ textAlign: "left", color: "darkblue", fontWeight: "bold" }}>Add New Employee</DialogTitle>
      <DialogContent>
        <Grid container spacing={1}>
          <Grid item xs={6}>
            <TextField
              fullWidth
              label="First Name"
              name="first_name"
              value={formData.first_name}
              onChange={handleChange}
              margin="normal"
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              fullWidth
              label="Last Name"
              name="last_name"
              value={formData.last_name}
              onChange={handleChange}
              margin="normal"
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              fullWidth
              label="Email"
              name="email"
              type="email"
              value={formData.email}
              onChange={handleChange}
              margin="normal"
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              fullWidth
              label="Phone"
              name="phone"
              value={formData.phone}
              onChange={handleChange}
              margin="normal"
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              fullWidth
              label="Nickname"
              name="nickname"
              value={formData.nickname}
              onChange={handleChange}
              margin="normal"
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              fullWidth
              label="Age"
              name="age"
              type="number"
              value={formData.age}
              onChange={handleChange}
              margin="normal"
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              fullWidth
              label="Address"
              name="address"
              value={formData.address}
              onChange={handleChange}
              margin="normal"
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              fullWidth
              label="District"
              name="district"
              value={formData.district}
              onChange={handleChange}
              margin="normal"
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              fullWidth
              label="Province"
              name="province"
              value={formData.province}
              onChange={handleChange}
              margin="normal"
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              fullWidth
              label="Jobposition"
              name="position"
              value={formData.position}
              onChange={handleChange}
              margin="normal"
            />
          </Grid>
          <Grid item xs={12}>
            <TextField
              fullWidth 
              label="Salary"
              name="salary"
              type="number"
              value={formData.salary}
              onChange={handleChange}
              margin="normal"
            />
          </Grid>
        </Grid>
      </DialogContent>
      <DialogActions>
        <Button onClick={handleClose}>Cancel</Button>
        <Button onClick={onSubmit} variant="contained" color="primary">
          Add Employee
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default EmployeeFormAlert;
