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
    const [userRole, setUserRole] = useState(''); // สถานะ role ของผู้ใช้

    useEffect(() => {
        setSelectedItem(location.pathname);
    }, [location]);

    useEffect(() => {
        // ตรวจสอบสถานะการเข้าสู่ระบบจาก localStorage
        const checkLoginStatus = () => {
            const loggedInUser = JSON.parse(localStorage.getItem("loggedInUser"));
            setIsLoggedIn(!!loggedInUser); // เปลี่ยนสถานะการเข้าสู่ระบบตามที่ได้จาก localStorage
            if (loggedInUser) {
                setUserRole(loggedInUser.role); // ดึง role ของผู้ใช้จากข้อมูลใน localStorage
            }
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
        // ลบข้อมูลผู้ใช้จาก localStorage และนำทางไปยังหน้าล็อกอิน
        localStorage.removeItem("loggedInUser");
        navigate('/login');
    };

    return (
        <DrawerStyled variant="permanent">
            <DrawerPaper>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', marginTop: 3 }}>
                    <Typography variant="h6">{isLoggedIn ? userRole : 'Guest'}</Typography> {/* แสดง role หรือ Guest หากยังไม่เข้าสู่ระบบ */}
                </Box>
                <Box sx={{ flexGrow: 1, marginTop: 1 }}>
                    {isLoggedIn ? ( // แสดงเมนูเฉพาะเมื่อเข้าสู่ระบบ
                        <List>
                            {[ 
                                { text: 'Dashboard', icon: <DashboardIcon />, path: '/dashboard' },
                                { text: 'Products', icon: <ShoppingCart />, path: '/products' },
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
                            {/* แสดงเมนู Employees เฉพาะเมื่อ role ไม่ใช่ employee */}
                            {userRole !== 'employee' && (
                                <ListItem
                                    button
                                    onClick={() => handleNavigation('/employees')}
                                    sx={{
                                        color: selectedItem === '/employees' ? 'blue' : 'inherit',
                                        '& .MuiListItemIcon-root': {
                                            color: selectedItem === '/employees' ? 'blue' : 'inherit',
                                        },
                                        borderLeft: selectedItem === '/employees' ? '4px solid blue' : 'none',
                                        cursor: 'pointer',
                                    }}
                                >
                                    <ListItemIcon><People /></ListItemIcon>
                                    <ListItemText primary="Employees" />
                                </ListItem>
                            )}
                        </List>
                    ) : (null)}
                </Box>
                <Box sx={{ marginBottom: 5 }}>
                    {!isLoggedIn && ( // แสดงปุ่ม Login หากยังไม่เข้าสู่ระบบ
                        <List>
                            <ListItem button key="Login" sx={{ cursor: 'pointer' }} onClick={() => navigate('/login')}>
                                <ListItemIcon><ExitToApp /></ListItemIcon>
                                <ListItemText primary="Login" />
                            </ListItem>
                        </List>
                    )}
                </Box>
                {isLoggedIn && ( // แสดงปุ่ม Logout เฉพาะเมื่อเข้าสู่ระบบ
                    <List>
                        <ListItem button key="Logout" sx={{ marginBottom: 5, cursor: 'pointer' }} onClick={handleLogout}>
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
