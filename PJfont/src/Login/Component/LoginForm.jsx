import React, { useState, useEffect } from "react";
import {
  TextField, Button, Paper, Typography, CircularProgress, Box,
} from "@mui/material";
import { useNavigate } from "react-router-dom";
import bcrypt from "bcryptjs";

const LoginForm = ({ onLogin }) => {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [Users, setUsers] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchUsers = async () => {
      try {
        const response = await fetch("http://localhost:8888/Users");
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        const data = await response.json();
        setUsers(data);
        console.log("Fetched Users:", data);
      } catch (error) {
        console.error("Error fetching Users:", error);
      }
    };
    fetchUsers();
  }, []);

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);

    const user = Users.find((user) => user.username === username);
    if (user) {
      const isPasswordValid = await bcrypt.compare(password, user.password_hash);
      if (isPasswordValid) {
        setError("");
        localStorage.setItem("loggedInUser", JSON.stringify(user));
        navigate("/");
      } else {
        setError("ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง");
      }
    } else {
      setError("ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง");
    }
    setLoading(false);
  };

  return (
    <Paper
      elevation={3}
      style={{ padding: "2rem", height: "400px", width: "350px" }}
    >
      <br />
      <br />
      <Typography variant="h5" component="h2" gutterBottomalign="center" sx={{ fontWeight: "bold" }}>
        Login into your account
      </Typography>
      <br />
      <br />
      <form onSubmit={handleLogin}>
        <Box display="flex" flexDirection="column">
          {" "}
          <TextField fullWidthlabel="Username" variant="outlined" value={username} onChange={(e) => setUsername(e.target.value)} />
          <br />
          <TextField fullWidthlabel="Password" type="password" variant="outlined" value={password} onChange={(e) => setPassword(e.target.value)} />
          <br />
          {error && <Typography color="error">{error}</Typography>}
          <br />
          <Button type="submit" fullWidthvariant="contained" color="primary" size="large" disabled={loading}>
            {loading ? (
              <CircularProgress size={24} color="inherit" />
            ) : (
              "Login now"
            )}
          </Button>
        </Box>
      </form>
    </Paper>
  );
};

export default LoginForm;
