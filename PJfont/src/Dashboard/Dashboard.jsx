import React, { useState, useEffect } from "react";
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
];

const historyData = [
    { prediction: 18000, sale: 17000, difference: -1000, percentageOfError: '5%' },
];

const productList = [
    { name: 'P1', price: 100 },
];

const dashboard = () => {
    const [Predictive, setPredictive] = useState([]);

    useEffect(() => {
        const fetchPredictives = async () => {
          try {
            const response = await fetch("http://localhost:8888/Predictive");
            if (!response.ok) {
              throw new Error("Network response was not ok");
            }
            const data = await response.json();
            setPredictive(data);
            console.log("Fetched Predictive:", data);
          } catch (error) {
            console.error("Error fetching Predictive:", error);
          }
        };
        fetchPredictives();
    }, []);

    return (
        <div style={{ display: 'flex' }}>
            <Sidebar />
            <div style={{ padding: '1px', flexGrow: 1, display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <Typography variant="h4" gutterBottom sx={{ textAlign: 'left', color: 'darkblue', fontWeight: 'bold' }}>Dashboard</Typography>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px' }}>
                    <SalesCard  title="Yesterday sales" amount="15,000 Bath" 
                                prediction="14,000 Bath" subtitle="Today's Prediction" 
                                Yessubtitle="Yesterday's Prediction" Yesprediction="15,000 Bath"/>
                    <PredictionCard title="Prediction for tomorrow" amount={`${!isNaN(Number(Predictive.predicted_sales)) ? Number(Predictive.predicted_sales).toFixed(2) : '0.00'} Bath`} accuracy={`${(100 - Predictive.percentage_error).toFixed(2)}%`}  />
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
