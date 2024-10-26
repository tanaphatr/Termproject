import React, { useState, useEffect } from "react";
import { Typography } from '@mui/material';
import Sidebar from '../Tool/Sidebar';
import SalesCard from './Component/SalesCard';
import PredictionCard from './Component/PredictionCard';
import WeatherCard from './Component/WeatherCard';
import SalesGraph from './Component/SalesGraph';
import ProductTable from './Component/ProductTable';
import HistoryTable from './Component/HistoryTable';

const historyData = [
    { prediction: 18000, sale: 17000, difference: -1000, percentageOfError: '5%' },
];

const Dashboard = () => {
    const [Predictive, setPredictive] = useState({});
    const [products, setProducts] = useState([]);
    const [Salesdata, setSalesdata] = useState([]);
    const [data, setData] = useState([]);
    const [temperature, setTemperature] = useState(null); // State สำหรับอุณหภูมิ
    const [weather, setWeather] = useState(null); // State สำหรับสภาพอากาศ
    const [tomorrowWeather, setTomorrowWeather] = useState({ temperature: null, condition: null }); // State สำหรับสภาพอากาศวันพรุ่งนี้
    const apiKey = '9f04441fb1254c3a8bf212302242009'; // แทนที่ YOUR_API_KEY ด้วย API Key ของคุณ
    const location = 'Bangkok'; // ตั้งค่าตำแหน่งที่ต้องการ

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

        const fetchProducts = async () => {
            try {
                const response = await fetch("http://localhost:8888/Products");
                if (!response.ok) {
                    throw new Error("Network response was not ok");
                }
                const data = await response.json();
                setProducts(data);
                console.log("Fetched Products:", data);
            } catch (error) {
                console.error("Error fetching Products:", error);
            }
        };

        const fetchSalesdata = async () => {
            try {
                const response = await fetch("http://localhost:8888/Salesdata");
                if (!response.ok) {
                    throw new Error("Network response was not ok");
                }
                const data = await response.json();

                // กรองข้อมูลให้เหลือ 60 ค่าสุดท้าย
                const latestData = data.slice(-60);
                setSalesdata(latestData);
                console.log("Fetched Salesdata:", latestData);

                // สร้างข้อมูลสำหรับกราฟ
                const graphData = latestData.map(item => ({
                    name: new Date(item.sale_date).toLocaleString('default', { month: 'short', day: 'numeric' }),
                    actual: item.sales_amount,
                    profit: item.profit_amount // ใช้ profit_amount จาก Salesdata
                }));

                setData(graphData); // ตั้งค่า data สำหรับกราฟ
            } catch (error) {
                console.error("Error fetching Salesdata:", error);
            }
        };

        const fetchWeather = async () => {
            try {
                const response = await fetch(`https://api.weatherapi.com/v1/forecast.json?key=${apiKey}&q=${location}&days=2`);
                if (!response.ok) {
                    throw new Error("Network response was not ok");
                }
                const weatherData = await response.json();
                const currentTemperature = weatherData.current.temp_c; // อุณหภูมิใน °C
                const weatherCondition = weatherData.current.condition.text; // สภาพอากาศ
                const tomorrowTemperature = weatherData.forecast.forecastday[1].day.avgtemp_c; // อุณหภูมิของวันพรุ่งนี้
                const tomorrowCondition = weatherData.forecast.forecastday[1].day.condition.text; // สภาพอากาศของวันพรุ่งนี้
                setTemperature(currentTemperature);
                setWeather(weatherCondition);
                setTomorrowWeather({ temperature: tomorrowTemperature, condition: tomorrowCondition });
                console.log("Fetched Weather:", currentTemperature, weatherCondition, tomorrowTemperature, tomorrowCondition);
            } catch (error) {
                console.error("Error fetching Weather:", error);
            }
        };

        fetchWeather();
        fetchProducts();
        fetchPredictives();
        fetchSalesdata();
    }, []);

    return (
        <div style={{ display: 'flex' }}>
            <Sidebar />
            <div style={{ padding: '1px', flexGrow: 1, display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <Typography variant="h4" gutterBottom sx={{ textAlign: 'left', color: 'darkblue', fontWeight: 'bold' }}>Dashboard</Typography>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px' }}>
                    <SalesCard title="Yesterday sales" amount="15,000 Bath"
                        prediction="14,000 Bath" subtitle="Today's Prediction"
                        Yessubtitle="Yesterday's Prediction" Yesprediction="15,000 Bath" />
                    <PredictionCard title="Prediction for tomorrow"
                        amount={`${!isNaN(Number(Predictive.predicted_sales)) ? Number(Predictive.predicted_sales).toFixed(2) : '0.00'} Bath`}
                        accuracy={`${(100 - Predictive.percentage_error).toFixed(2)}%`} />
                    <WeatherCard title="Weather for Tomorrow"
                        temperature={tomorrowWeather.temperature ? `${tomorrowWeather.temperature} °C` : 'Loading...'}
                        weather={tomorrowWeather.condition ? tomorrowWeather.condition : 'Loading...'}
                        date={new Date(Date.now() + 86400000).toLocaleDateString()} /> {/* แสดงวันที่เป็นวันพรุ่งนี้ */}
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px' }}>
                    <SalesGraph data={data} />
                    <ProductTable products={products} />
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px' }}>
                    <HistoryTable historyData={historyData} />
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
