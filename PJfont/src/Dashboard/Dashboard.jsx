import React, { useState, useEffect } from "react";
import { Typography } from '@mui/material';
import Sidebar from '../Tool/Sidebar';
import SalesCard from './Component/SalesCard';
import PredictionCard from './Component/PredictionCard';
import WeatherCard from './Component/WeatherCard';
import SalesGraph from './Component/SalesGraph';
import ProductTable from './Component/ProductTable';
import HistoryTable from './Component/HistoryTable';

const Dashboard = () => {
    const [Predictive, setPredictive] = useState({});
    const [products, setProducts] = useState([]);
    const [Salesdata, setSalesdata] = useState([]);
    const [Salesprediction, setSalesprediction] = useState([]);
    const [data, setData] = useState([]);
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

        const fetchSalesprediction = async () => {
            try {
                const response = await fetch("http://localhost:8888/Sales_prediction");
                if (!response.ok) {
                    throw new Error("Network response was not ok");
                }
                const data = await response.json();
                setSalesprediction(data);
                console.log("Fetched Salesprediction:", data);
            } catch (error) {
                console.error("Error fetching Salesprediction:", error);
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
                setSalesdata(data);
        
                const latestData = data.slice(-300);
                console.log("Fetched Salesdata:", latestData);
        
                // สร้างข้อมูลสำหรับกราฟ
                const graphData = latestData.map(item => ({
                    name: new Date(item.sale_date).toLocaleString('default', { month: 'short', year: 'numeric' }), // แยกปีและเดือน
                    actual: isNaN(Number(item.sales_amount)) ? 0 : Number(item.sales_amount),  // แปลง sales_amount เป็นตัวเลข ถ้าไม่ใช่จะใช้ค่า 0
                    profit: isNaN(Number(item.profit_amount)) ? 0 : Number(item.profit_amount)   // แปลง profit_amount เป็นตัวเลข ถ้าไม่ใช่จะใช้ค่า 0
                }));
        
                // รวมยอดขายและกำไรตามเดือน
                const monthlyData = graphData.reduce((acc, item) => {
                    if (!acc[item.name]) {
                        acc[item.name] = { name: item.name, actual: 0, profit: 0 };
                    }
                    acc[item.name].actual += item.actual;  // รวมยอดขาย
                    acc[item.name].profit += item.profit;  // รวมกำไร
                    return acc;
                }, {});
        
                // แปลงผลลัพธ์ให้เป็นอาร์เรย์
                const finalGraphData = Object.values(monthlyData);
        
                setData(finalGraphData); // ตั้งค่า data สำหรับกราฟ
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
                const tomorrowTemperature = weatherData.forecast.forecastday[1].day.maxtemp_c; // อุณหภูมิของวันพรุ่งนี้
                const tomorrowCondition = weatherData.forecast.forecastday[1].day.condition.text; // สภาพอากาศของวันพรุ่งนี้
                setTomorrowWeather({ temperature: tomorrowTemperature, condition: tomorrowCondition });
                console.log("Fetched Weather:", tomorrowTemperature, tomorrowCondition);
            } catch (error) {
                console.error("Error fetching Weather:", error);
            }
        };

        fetchWeather();
        fetchSalesprediction();
        fetchProducts();
        fetchPredictives();
        fetchSalesdata();
    }, []);
    console.log(Predictive.predictions);
    const filteredProducts = Predictive.predictions
    ? Predictive.predictions.filter((item) => item.Product_code && item.Prediction)
    : []; // ถ้า Predictive.predictions เป็น undefined ให้ใช้อาเรย์ว่าง
      
    return (
        <div style={{ display: 'flex' }}>
            <Sidebar />
            <div style={{ padding: '1px', flexGrow: 1, display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <Typography variant="h4" gutterBottom sx={{ textAlign: 'left', color: 'darkblue', fontWeight: 'bold' }}>Dashboard</Typography>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px' }}>
                    <SalesCard
                        title="Yesterday sales"
                        amount={`${Salesdata.length > 0 ? Salesdata[Salesdata.length - 1].sales_amount : 0} Bath`} // แสดงยอดขายล่าสุด
                        Yessubtitle="Yesterday's Prediction"
                        Yesprediction={`${Salesprediction.length > 0 ? Salesprediction[0].predicted_sales : 0} Bath`} />
                    <PredictionCard title="Prediction for Today"
                        amount={`${!isNaN(Number(Predictive.predicted_sales)) ? Number(Predictive.predicted_sales).toFixed(2) : '0.00'} Bath`}
                        accuracy={Predictive.predicted_date} />
                    <WeatherCard title="Weather for Tomorrow"
                        temperature={tomorrowWeather.temperature ? `${tomorrowWeather.temperature} °C` : 'Loading...'}
                        weather={tomorrowWeather.condition ? tomorrowWeather.condition : 'Loading...'}
                        date={new Date(Date.now() + 86400000).toLocaleDateString()} /> {/* แสดงวันที่เป็นวันพรุ่งนี้ */}
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px' }}>
                    <SalesGraph data={data} />
                    <ProductTable products={filteredProducts} mount={Predictive.predictions}/>
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px' }}>
                    <HistoryTable historyData={Salesprediction} />
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
