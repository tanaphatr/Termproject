import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';

const SalesCard = ({ title, amount, prediction, subtitle }) => {
    return (
        <Card style={{ flex: '1 1 calc(40% - 16px)' }}>
            <CardContent>
                <Typography variant="h6">{title}</Typography>
                <Typography variant="h4">{amount}</Typography>
                <br />
                <Typography variant="body2" color="textSecondary">{subtitle}: {prediction}</Typography>
            </CardContent>
        </Card>
    );
};

export default SalesCard;
