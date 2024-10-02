const express = require('express');
const router = express.Router();

router.get('/', (req, res) => {
    res.send(`
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Server</title>
            <script>
                function navigateTo(path) {
                    window.location.href = path;
                }
            </script>
            <style>
                body {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    padding-top: 30px; 
                }
                .container {
                    text-align: center;
                }
            </style>
        </head>
        <body class="container">

        <button onclick="navigateTo('/Users/html')">Users</button>
        <button onclick="navigateTo('/Products/html')">Products</button>
        <button onclick="navigateTo('/Employees/html')">Employees</button>
        <button onclick="navigateTo('/Daily_salesRoutes/html')">Daily_salesRoutes</button>
        <button onclick="navigateTo('/Product_sales/html')">Product_sales</button>
        <button onclick="navigateTo('/Product_stock_history/html')">Product_stock_history</button>
        <button onclick="navigateTo('/Sales_prediction/html')">Sales_prediction</button>
        <button onclick="navigateTo('/Salesdata/html')">Salesdata</button>
        </body>
        </html>
    `);
});

module.exports = router;