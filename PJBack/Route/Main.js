const express = require('express');
const router = express.Router();
router.get('/', (req, res) => {
    res.send(`
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ระบบจัดการข้อมูล</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <script>
                function navigateTo(path) {
                    window.location.href = path;
                }
                function applyHoverEffect(button) {
                    button.classList.add('bg-blue-600', 'scale-105');
                }
                function removeHoverEffect(button) {
                    button.classList.remove('bg-blue-600', 'scale-105');
                }
            </script>
            <style>
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(-20px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                .fade-in {
                    animation: fadeIn 0.5s ease-out forwards;
                }
            </style>
        </head>
        <body class="bg-gradient-to-r from-blue-100 to-purple-100 min-h-screen flex items-center justify-center">
            <div class="container mx-auto px-4 py-8">
                <h1 class="text-4xl font-bold text-center text-gray-800 mb-8 fade-in">ระบบจัดการข้อมูล</h1>
                <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                    <button onclick="navigateTo('/Users/html')" class="bg-blue-500 hover:bg-blue-600 text-white font-bold py-4 px-6 rounded-lg shadow-md transition duration-300 ease-in-out transform hover:scale-105 fade-in" style="animation-delay: 0.1s;" onmouseover="applyHoverEffect(this)" onmouseout="removeHoverEffect(this)">Users</button>
                    <button onclick="navigateTo('/Products/html')" class="bg-green-500 hover:bg-green-600 text-white font-bold py-4 px-6 rounded-lg shadow-md transition duration-300 ease-in-out transform hover:scale-105 fade-in" style="animation-delay: 0.2s;" onmouseover="applyHoverEffect(this)" onmouseout="removeHoverEffect(this)">Products</button>
                    <button onclick="navigateTo('/Employees/html')" class="bg-yellow-500 hover:bg-yellow-600 text-white font-bold py-4 px-6 rounded-lg shadow-md transition duration-300 ease-in-out transform hover:scale-105 fade-in" style="animation-delay: 0.3s;" onmouseover="applyHoverEffect(this)" onmouseout="removeHoverEffect(this)">Employees</button>
                    <button onclick="navigateTo('/Daily_salesRoutes/html')" class="bg-red-500 hover:bg-red-600 text-white font-bold py-4 px-6 rounded-lg shadow-md transition duration-300 ease-in-out transform hover:scale-105 fade-in" style="animation-delay: 0.4s;" onmouseover="applyHoverEffect(this)" onmouseout="removeHoverEffect(this)">Daily Sales</button>
                    <button onclick="navigateTo('/Product_sales/html')" class="bg-indigo-500 hover:bg-indigo-600 text-white font-bold py-4 px-6 rounded-lg shadow-md transition duration-300 ease-in-out transform hover:scale-105 fade-in" style="animation-delay: 0.5s;" onmouseover="applyHoverEffect(this)" onmouseout="removeHoverEffect(this)">Product Sales</button>
                    <button onclick="navigateTo('/Product_stock_history/html')" class="bg-purple-500 hover:bg-purple-600 text-white font-bold py-4 px-6 rounded-lg shadow-md transition duration-300 ease-in-out transform hover:scale-105 fade-in" style="animation-delay: 0.6s;" onmouseover="applyHoverEffect(this)" onmouseout="removeHoverEffect(this)">Product Stock History</button>
                    <button onclick="navigateTo('/Sales_prediction/html')" class="bg-pink-500 hover:bg-pink-600 text-white font-bold py-4 px-6 rounded-lg shadow-md transition duration-300 ease-in-out transform hover:scale-105 fade-in" style="animation-delay: 0.7s;" onmouseover="applyHoverEffect(this)" onmouseout="removeHoverEffect(this)">Sales Prediction</button>
                    <button onclick="navigateTo('/Salesdata/html')" class="bg-teal-500 hover:bg-teal-600 text-white font-bold py-4 px-6 rounded-lg shadow-md transition duration-300 ease-in-out transform hover:scale-105 fade-in" style="animation-delay: 0.8s;" onmouseover="applyHoverEffect(this)" onmouseout="removeHoverEffect(this)">Sales Data</button>
                </div>
            </div>
        </body>
        </html>
    `);
});
module.exports = router;