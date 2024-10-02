import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';

const WeatherCard = ({ title, temperature }) => {
    return (
        <Card style={{ flex: '1 1 calc(20% - 16px)' }}>
            <CardContent>
                <Typography variant="h6">{title}</Typography>
                <br />
                <Typography variant="h4">{temperature}</Typography>
            </CardContent>
        </Card>
    );
};

export default WeatherCard;
