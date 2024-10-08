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

    useEffect(() => {
        setSelectedItem(location.pathname);
    }, [location]);

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

    return (
        <DrawerStyled variant="permanent">
            <DrawerPaper>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', marginTop: 3 }}>
                    <Typography variant="h6">Name</Typography>
                </Box>
                <Box sx={{ flexGrow: 1, marginTop: 1 }}>
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
                                    '&:hover': {
                                    },
                                    borderLeft: selectedItem === path ? '4px solid blue' : 'none',
                                }}
                            >
                                <ListItemIcon>{icon}</ListItemIcon>
                                <ListItemText primary={text} />
                            </ListItem>
                        ))}
                    </List>
                </Box>
                <List>
                    <ListItem button key="Logout" sx={{ marginBottom: 5 }}>
                        <ListItemIcon><ExitToApp /></ListItemIcon>
                        <ListItemText primary="Logout" />
                    </ListItem>
                </List>
            </DrawerPaper>
        </DrawerStyled>
    );
}

export default Sidebar;