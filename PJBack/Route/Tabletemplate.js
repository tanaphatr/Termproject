function generateHtmlPage(title, fields, rows) {
  return `
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>${title}</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <style>
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(-20px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                .fade-in {
                    animation: fadeIn 0.5s ease-out forwards;
                }
                .table-cell {
                    overflow-wrap: break-word; /* ทำให้ข้อความยาวถูกตัดบรรทัด */
                    max-width: 150px; /* กำหนดความกว้างสูงสุด */
                }
            </style>
        </head>
        <body class="bg-gradient-to-r from-blue-100 to-purple-100 min-h-screen flex flex-col items-center pt-8">
            <h1 class="text-3xl font-bold text-center text-gray-800 mb-8 fade-in">${title}</h1>
            <div class="button-container grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
                <button onclick="navigateTo('/Users/html')" class="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out transform hover:scale-105 fade-in">Users</button>
                <button onclick="navigateTo('/Products/html')" class="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out transform hover:scale-105 fade-in">Products</button>
                <button onclick="navigateTo('/Employees/html')" class="bg-yellow-500 hover:bg-yellow-600 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out transform hover:scale-105 fade-in">Employees</button>
                <button onclick="navigateTo('/Daily_salesRoutes/html')" class="bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out transform hover:scale-105 fade-in">Daily Sales</button>
                <button onclick="navigateTo('/Product_sales/html')" class="bg-indigo-500 hover:bg-indigo-600 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out transform hover:scale-105 fade-in">Product Sales</button>
                <button onclick="navigateTo('/Product_stock_history/html')" class="bg-purple-500 hover:bg-purple-600 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out transform hover:scale-105 fade-in">Product Stock History</button>
                <button onclick="navigateTo('/Sales_prediction/html')" class="bg-pink-500 hover:bg-pink-600 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out transform hover:scale-105 fade-in">Sales Prediction</button>
                <button onclick="navigateTo('/Salesdata/html')" class="bg-teal-500 hover:bg-teal-600 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out transform hover:scale-105 fade-in">Sales Data</button>
            </div>

            <div class="w-[100%] px-4 overflow-x-auto">
                <table class="w-full bg-white shadow-md rounded-lg overflow-hidden border-collapse">
                    <thead class="bg-gray-200">
                        <tr>
                            ${fields
                              .map(
                                (field) => `
                                <th class="py-3 px-4 text-left font-semibold text-gray-700 border border-gray-300">${field.name}</th>
                            `
                              )
                              .join("")}
                            <th class="py-3 px-4 text-left font-semibold text-gray-700 border border-gray-300">Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${rows
                          .map(
                            (row, index) => `
                            <tr class="${
                              index % 2 === 0 ? "bg-gray-50" : "bg-white"
                            }">
                                ${fields
                                  .map(
                                    (field) => `
                                    <td class="py-3 px-4 border border-gray-300 table-cell">${
                                      row[field.name] !== undefined
                                        ? row[field.name]
                                        : ""
                                    }</td>
                                `
                                  )
                                  .join("")}
                                <td class="py-3 px-4 border flex flex-col space-y-2" style="height: 50%;">
                                    <button onclick="editRow(${index})" class="bg-blue-500 hover:bg-blue-600 text-white font-bold py-1 px-2 rounded">Edit</button>   
                                </td>
                                <td class="py-3 px-4 border flex flex-col space-y-2" style="height: 50%;">
                                    <button onclick="deleteRow(${index})" class="bg-red-500 hover:bg-red-600 text-white font-bold py-1 px-2 rounded">Delete</button>
                                </td>
                            </tr>
                        `
                          )
                          .join("")}
                    </tbody>
                </table>
            </div>

            <script>
                function navigateTo(url) {
                    window.location.href = url;
                }

                function editRow(index) {
                    console.log('Edit row:', index);
                    alert('Edit functionality not implemented yet');
                }

                function deleteRow(index) {
                    console.log('Delete row:', index);
                    alert('Delete functionality not implemented yet');
                }
            </script>
        </body>
        </html>
    `;
}

module.exports = generateHtmlPage;
