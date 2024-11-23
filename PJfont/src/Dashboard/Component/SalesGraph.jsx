import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card, CardContent, Typography } from '@mui/material';

const SalesGraph = ({ data }) => {
    return (
        <Card style={{ flex: '1 1 calc(80% - 16px)' }}>
            <CardContent>
                <Typography variant="h6">Sales Graph</Typography>
                <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis 
                            ticks={[0, 150000, 300000, 450000, 600000]} // กำหนดค่าที่จะแสดงในแกน Y
                            domain={[0, 600000]} // กำหนดช่วงของแกน Y
                        />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey="actual" stroke="#8884d8" />
                        <Line type="monotone" dataKey="profit" stroke="#82ca9d" />
                    </LineChart>
                </ResponsiveContainer>
            </CardContent>
        </Card>
    );
};

export default SalesGraph;
