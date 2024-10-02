import React, { useState } from 'react';
import {
  TextField,
  Button,
  Checkbox,
  FormControlLabel,
  Paper,
  Typography,
  Box,
} from '@mui/material';
import { Email, Lock } from '@mui/icons-material';

const LoginForm = ({ onLogin }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(false);

  const handleLogin = (e) => {
    e.preventDefault();
    onLogin({ email, password, rememberMe });
  };

  return (
    <Paper elevation={3} style={{ padding: '2rem', height: '450px' ,width: '350px'}}>
      <Typography variant="h5" component="h2" gutterBottom>
        Login into your account
      </Typography>
      <br/>
      <br/>
      <br/>
      <form onSubmit={handleLogin}>
        <TextField
          fullWidth
          margin="normal"
          label="Email"
          variant="outlined"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          InputProps={{
            endAdornment: <Email color="primary" />,
          }}
        />
        <br/>
        <br/>
        <TextField
          fullWidth
          margin="normal"
          label="Password"
          type="password"
          variant="outlined"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          InputProps={{
            endAdornment: <Lock color="primary" />,
          }}
        />
        <br/>
        <Box display="flex" alignItems="center" marginY={2}>
          <FormControlLabel
            control={
              <Checkbox
                checked={rememberMe}
                onChange={(e) => setRememberMe(e.target.checked)}
                color="primary"
              />
            }
            label="Remember me"
          />
        </Box>
        <Button
          type="submit"
          fullWidth
          variant="contained"
          color="primary"
          size="large"
          style={{ marginTop: '1rem' }}
        >
          Login now
        </Button>
      </form>
    </Paper>
  );
};

export default LoginForm;
