import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LabelList ,BarChart, Bar} from 'recharts';
import { Card, CardContent, Typography } from '@mui/material';

const SalesGraph = ({ data }) => {
    return (
        <Card style={{ flex: '1 1 calc(80% - 16px)' }}>
            <CardContent>
                <Typography variant="h6">Sales Graph</Typography>
                <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis
                            ticks={[0, 150000, 300000, 450000, 600000]} // กำหนดค่าที่จะแสดงในแกน Y
                            domain={[0, 600000]} // กำหนดช่วงของแกน Y
                        />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="Sales" fill="#8884d8">
                            <LabelList dataKey="Sales" position="top" />
                        </Bar>
                        <Bar dataKey="profit" fill="#82ca9d">
                            <LabelList dataKey="profit" position="top" />
                        </Bar>
                        {/* <Bar dataKey="predic" fill="#82ca9d">
                            <LabelList dataKey="predic" position="top" />
                        </Bar> */}
                    </BarChart>

                </ResponsiveContainer>
            </CardContent>
        </Card>
    );
};

export default SalesGraph;
