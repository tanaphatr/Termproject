import React, { useState, useEffect } from 'react';
import { styled } from '@mui/material/styles';
import { Drawer, List, ListItem, ListItemIcon, ListItemText, Box, Typography } from '@mui/material';
import { Dashboard as DashboardIcon, ShoppingCart, People, Assessment, ExitToApp } from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';

function Sidebar() {
    const drawerWidth = 220;
    const navigate = useNavigate();
    const location = useLocation();
    const [selectedItem, setSelectedItem] = useState('/dashboard');
    const [isLoggedIn, setIsLoggedIn] = useState(false); // สถานะการเข้าสู่ระบบ

    useEffect(() => {
        setSelectedItem(location.pathname);
    }, [location]);

    useEffect(() => {
        // ตรวจสอบสถานะการเข้าสู่ระบบ (ปรับตามที่คุณมีในแอปของคุณ)
        const checkLoginStatus = () => {
            // แทนที่ด้วยการตรวจสอบการเข้าสู่ระบบที่แท้จริง
            const loggedIn = true; // เปลี่ยนให้เป็นการตรวจสอบที่แท้จริง
            setIsLoggedIn(loggedIn);
        };

        checkLoginStatus();
    }, []);

    const DrawerStyled = styled(Drawer)(() => ({
        width: drawerWidth,
        flexShrink: 0,
    }));

    const DrawerPaper = styled('div')(() => ({
        width: drawerWidth,
        boxSizing: 'border-box',
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
    }));

    const handleNavigation = (path) => {
        setSelectedItem(path);
        navigate(path);
    };

    const handleLogout = () => {
        // ที่นี่คุณอาจจะต้องทำการล้างข้อมูลผู้ใช้ (เช่น token) ก่อนนำทาง
        navigate('/login'); // นำทางไปยัง /login
    };

    return (
        <DrawerStyled variant="permanent">
            <DrawerPaper>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', marginTop: 3 }}>
                    <Typography variant="h6">Name</Typography>
                </Box>
                <Box sx={{ flexGrow: 1, marginTop: 1 }}>
                    {isLoggedIn && ( // แสดงเมนูเฉพาะเมื่อเข้าสู่ระบบ
                        <List>
                            {[
                                { text: 'Dashboard', icon: <DashboardIcon />, path: '/dashboard' },
                                { text: 'Products', icon: <ShoppingCart />, path: '/products' },
                                { text: 'Employees', icon: <People />, path: '/employees' },
                                { text: 'Reports', icon: <Assessment />, path: '/reports' },
                            ].map(({ text, icon, path }) => (
                                <ListItem
                                    button
                                    key={text}
                                    onClick={() => handleNavigation(path)}
                                    sx={{
                                        color: selectedItem === path ? 'blue' : 'inherit',
                                        '& .MuiListItemIcon-root': {
                                            color: selectedItem === path ? 'blue' : 'inherit',
                                        },
                                        borderLeft: selectedItem === path ? '4px solid blue' : 'none',
                                        cursor: 'pointer',
                                    }}
                                >
                                    <ListItemIcon>{icon}</ListItemIcon>
                                    <ListItemText primary={text} />
                                </ListItem>
                            ))}
                        </List>
                    )}
                </Box>
                {isLoggedIn && ( // แสดงปุ่ม Logout เฉพาะเมื่อเข้าสู่ระบบ
                    <List>
                        <ListItem button key="Logout" sx={{ marginBottom: 5 , cursor: 'pointer', }} onClick={handleLogout}>
                            <ListItemIcon><ExitToApp /></ListItemIcon>
                            <ListItemText primary="Logout" />
                        </ListItem>
                    </List>
                )}
            </DrawerPaper>
        </DrawerStyled>
    );
}

export default Sidebar;
