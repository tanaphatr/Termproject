const express = require('express');
const cors = require('cors');

const mainRoutes = require('./Route/Main'); // Use consistent naming convention
const UsersRoutes = require('./Route/Table/Users')
const ProductsRoutes = require('./Route/Table/Products')
const EmployeesRoutes = require('./Route/Table/Employees')
const Daily_salesRoutes = require('./Route/Table/Daily_sales')
const Product_sales = require('./Route/Table/Product_sales')
const Product_stock_history = require('./Route/Table/Product_stock_history')
const Sales_prediction = require('./Route/Table/Sales_prediction')
const Salesdata = require('./Route/Table/Salesdata')
const Predictive = require('./Route/Table/ML/Predictive')

const app = express();
const hostname = 'http://localhost';
const port = 8888;

// Middleware
app.use(express.json()); // Parse JSON request bodies
app.use(cors()); // Enable CORS for all requests

// Main Server
app.use('/', mainRoutes);
// Use routes from the new files
app.use('/Users', UsersRoutes);
app.use('/Products', ProductsRoutes);
app.use('/Employees', EmployeesRoutes);
app.use('/Daily_salesRoutes', Daily_salesRoutes);
app.use('/Product_sales', Product_sales);
app.use('/Product_stock_history', Product_stock_history);
app.use('/Sales_prediction', Sales_prediction);
app.use('/Salesdata', Salesdata);
app.use('/Predictive', Predictive);


// Start the server
app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
    console.log(`Server is running on \x1b[31m${hostname}:${port}\x1b[0m`);
});

// Main Server
app.use('/', mainRoutes);