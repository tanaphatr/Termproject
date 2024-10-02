function generateHtmlPage(title, fields, rows) {
    return `
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>${title}</title>
            <style>
                body {
                    display: flex;
                    align-items: center;
                    flex-direction: column;
                    height: 100vh;
                    margin: 0;
                    padding-top: 30px;
                    font-family: Arial, sans-serif;
                }
                .button-container {
                    margin-bottom: 20px;
                }
                table {
                    border-collapse: collapse;
                    width: 80%;
                    margin: 20px 0;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 8px;
                }
                th {
                    background-color: #f2f2f2;
                    text-align: center;
                }
            </style>
        </head>
        <body>
            <div class="button-container">
                <button onclick="navigateTo('/Users/html')">Users</button>
                <button onclick="navigateTo('/Products/html')">Products</button>
                <button onclick="navigateTo('/Employees/html')">Employees</button>
                <button onclick="navigateTo('/Daily_salesRoutes/html')">Daily_salesRoutes</button>
                <button onclick="navigateTo('/Product_sales/html')">Product_sales</button>
                <button onclick="navigateTo('/Product_stock_history/html')">Product_stock_history</button>
                <button onclick="navigateTo('/Sales_prediction/html')">Sales_prediction</button>
                <button onclick="navigateTo('/Salesdata/html')">Salesdata</button>
            </div>

            <table>
                <thead>
                    <tr>
                        ${fields.map(field => `<th>${field.name}</th>`).join('')}
                    </tr>
                </thead>
                <tbody>
                    ${rows.map(row => `
                        <tr>
                            ${fields.map(field => `<td>${row[field.name]}</td>`).join('')}
                        </tr>
                    `).join('')}
                </tbody>
            </table>

            <script>
                function navigateTo(url) {
                    window.location.href = url;
                }
            </script>
        </body>
        </html>
    `;
}

module.exports = generateHtmlPage;
