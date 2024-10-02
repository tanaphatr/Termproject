import React from 'react';
import { Typography } from '@mui/material';
import Sidebar from '../Tool/Sidebar';
import SalesCard from './Component/SalesCard';
import PredictionCard from './Component/PredictionCard';
import WeatherCard from './Component/WeatherCard';
import SalesGraph from './Component/SalesGraph';
import ProductTable from './Component/ProductTable';
import HistoryTable from './Component/HistoryTable';

const data = [
    { name: 'Jan', actual: 4000, prediction: 2400 },
    { name: 'Feb', actual: 3000, prediction: 1398 },
    { name: 'Mar', actual: 2000, prediction: 9800 },
    { name: 'Apr', actual: 2780, prediction: 3908 },
    { name: 'May', actual: 1890, prediction: 4800 },
    { name: 'Jun', actual: 2390, prediction: 3800 },
];

const historyData = [
    { prediction: 18000, sale: 17000, difference: -1000, percentageOfError: '5%' },
    { prediction: 19000, sale: 17500, difference: -1500, percentageOfError: '7%' },
    { prediction: 20000, sale: 19000, difference: -1000, percentageOfError: '5%' },
];

const productList = [
    { name: 'P1', price: 100 },
    { name: 'P2', price: 200 },
    { name: 'P3', price: 150 },
    { name: 'P4', price: 300 },
    { name: 'P5', price: 250 },
];

const dashboard = () => {
    return (
        <div style={{ display: 'flex' }}>
            <Sidebar />
            <div style={{ padding: '1px', flexGrow: 1, display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <Typography variant="h4" gutterBottom sx={{ textAlign: 'left', color: 'darkblue', fontWeight: 'bold' }}>Dashboard</Typography>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px' }}>
                    <SalesCard title="Today sales" amount="15,000 Bath" prediction="14,000 Bath" subtitle="Today's Prediction" />
                    <PredictionCard title="Prediction for tomorrow" amount="17,000 Bath" accuracy="70%" />
                    <WeatherCard title="Weather tomorrow" temperature="50" />
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px' }}>
                    <SalesGraph data={data} />
                    <ProductTable products={productList} />
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px' }}>
                    <HistoryTable historyData={historyData} />
                </div>
            </div>
        </div>
    );
};

export default dashboard;
