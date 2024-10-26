import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';

const WeatherCard = ({ title, temperature ,weather ,date}) => {
    return (
        <Card style={{ flex: '1 1 calc(20% - 16px)' }}>
            <CardContent>
                <Typography variant="h6">{title}</Typography>
                <br />
                <Typography variant="h4">{temperature}</Typography>
                <Typography variant="h7">{weather}</Typography>
                <Typography variant="h6">{date}</Typography>
            </CardContent>
        </Card>
    );
};

export default WeatherCard;
