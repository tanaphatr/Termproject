function generateHtmlPage(title, fields, rows) {
    return `
          <!DOCTYPE html>
          <html lang="en">
          <head>
              <meta charset="UTF-8">
              <meta name="viewport" content="width=device-width, initial-scale=1.0">
              <title>${title}</title>
              <script src="https://cdn.tailwindcss.com"></script>
              <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
              <style>
                  @keyframes fadeIn {
                      from { opacity: 0; transform: translateY(-20px); }
                      to { opacity: 1; transform: translateY(0); }
                  }
                  .fade-in {
                      animation: fadeIn 0.5s ease-out forwards;
                  }
                  .table-cell {
                      overflow-wrap: break-word;
                      max-width: 150px;
                  }
              </style>
          </head>
          <body class="bg-gradient-to-r from-blue-100 to-purple-100 min-h-screen flex flex-col items-center pt-8">
              <h1 class="text-3xl font-bold text-center text-gray-800 mb-8 fade-in">${title}</h1>
              <div class="button-container grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
                  <button onclick="navigateTo('/Users/html')" class="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out transform hover:scale-105 fade-in">Users</button>
                  <button onclick="navigateTo('/Products/html')" class="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out transform hover:scale-105 fade-in">Products</button>
                  <button onclick="navigateTo('/Employees/html')" class="bg-yellow-500 hover:bg-yellow-600 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out transform hover:scale-105 fade-in">Employees</button>
                  <button onclick="navigateTo('/Daily_sales/html')" class="bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out transform hover:scale-105 fade-in">Daily Sales</button>
                  <button onclick="navigateTo('/Product_sales/html')" class="bg-indigo-500 hover:bg-indigo-600 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out transform hover:scale-105 fade-in">Product Sales</button>
                  <button onclick="navigateTo('/Product_stock_history/html')" class="bg-purple-500 hover:bg-purple-600 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out transform hover:scale-105 fade-in">Product Stock History</button>
                  <button onclick="navigateTo('/Sales_prediction/html')" class="bg-pink-500 hover:bg-pink-600 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out transform hover:scale-105 fade-in">Sales Prediction</button>
                  <button onclick="navigateTo('/Salesdata/html')" class="bg-teal-500 hover:bg-teal-600 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out transform hover:scale-105 fade-in">Sales Data</button>
              </div>
              <button onclick="openAddModal()" class="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded-lg shadow-md mb-4">Add New Record</button>
  
              <div class="w-[100%] px-4 overflow-x-auto">
                  <table class="w-full bg-white shadow-md rounded-lg overflow-hidden border-collapse">
                      <thead class="bg-gray-200">
                          <tr>
                              ${fields
                                .map(
                                  (field) =>
                                    `<th class="py-3 px-4 border border-gray-300 table-cell">${field.name}</th>`
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
                                      (field) =>
                                        `<td class="py-3 px-4 border border-gray-300 table-cell">${
                                          row[field.name] !== undefined && row[field.name] !== null
                                            ? row[field.name]
                                            : ""
                                        }</td>`
                                    )
                                    .join("")}
                                  <td class="py-3 px-4 border border-gray-300">
                                      <button onclick="openEditModal(${
                                        row.sales_data_id
                                      })" class="bg-blue-500 hover:bg-blue-600 text-white font-bold py-1 px-2 rounded mr-2">Edit</button>
                                      <button onclick="deleteRow(${
                                        row.sales_data_id
                                      })" class="bg-red-500 hover:bg-red-600 text-white font-bold py-1 px-2 rounded">Delete</button>
                                  </td>
                              </tr>
                          `
                            )
                            .join("")}
                      </tbody>
                  </table>
              </div>
            <!-- Add Modal -->
            <div id="addModal" class="hidden fixed inset-0 z-50 flex items-center justify-center bg-gray-800 bg-opacity-50">
                <div class="bg-white p-6 rounded-lg shadow-lg max-w-3xl w-full">
                    <h2 class="text-2xl font-semibold text-center mb-4">Add New Data</h2>
                    <form id="addForm" class="flex flex-wrap">
                    ${fields
                        .filter(field => !['sales_data_id', 'day_of_week', 'Seasonal'].includes(field.name))
                        .map(
                        (field) => `
                            <div class="mb-4 w-1/2 pr-2">
                                <label class="block text-sm font-medium text-gray-700 mb-1">${field.name}:</label>
                                <input type="${field.name === 'sale_date' ? 'text' : field.name.toLowerCase().includes('date') ? 'date' : 'text'}" 
                                        name="${field.name}" 
                                        id="add_${field.name}" 
                                        class="border border-gray-300 p-2 rounded-md w-full focus:outline-none focus:ring-2 focus:ring-blue-500" 
                                        ${!['sales_amount', 'profit_amount'].includes(field.name) ? 'required' : ''} />
                            </div>
                        `
                        )
                        .join("")}                                           
                        <div class="flex justify-between w-full mt-4">
                            <button type="submit" class="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded-lg shadow">Add Record</button>
                            <button type="button" onclick="closeAddModal()" class="bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded-lg shadow">Cancel</button>
                        </div>
                    </form>
                </div>
            </div>

            <!-- Edit Modal -->
            <div id="editModal" class="hidden fixed inset-0 z-50 flex items-center justify-center bg-gray-800 bg-opacity-50">
                <div class="bg-white p-6 rounded-lg shadow-lg max-w-3xl w-full">
                    <h2 class="text-2xl font-semibold text-center mb-4">Edit Data</h2>
                    <form id="editForm" class="flex flex-wrap">
                    ${fields
                        .filter(field => !['sales_data_id', 'day_of_week', 'Seasonal'].includes(field.name))
                        .map(
                        (field) => `
                            <div class="mb-4 w-1/2 pr-2">
                                <label class="block text-sm font-medium text-gray-700 mb-1">${field.name}:</label>
                                <input type="${field.name === 'sale_date' ? 'text' : field.name.toLowerCase().includes('date') ? 'date' : 'text'}" 
                                        name="${field.name}" 
                                        id="edit_${field.name}" 
                                        class="border border-gray-300 p-2 rounded-md w-full focus:outline-none focus:ring-2 focus:ring-blue-500" 
                                        ${!['sales_amount', 'profit_amount'].includes(field.name) ? 'required' : ''} />
                            </div>
                        `
                        )
                        .join("")}                                           
                        <div class="flex justify-between w-full mt-4">
                            <button type="submit" class="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-lg shadow">Update Record</button>
                            <button type="button" onclick="closeEditModal()" class="bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded-lg shadow">Cancel</button>
                        </div>
                    </form>
                </div>
            </div>
  
              <script>
                  const rowsData = ${JSON.stringify(rows)};
                  let currentSalesDataId;
  
                  function navigateTo(url) {
                      window.location.href = url;
                  }
  
                  function openAddModal() {
                      document.getElementById('addModal').classList.remove('hidden');
                  }
  
                  function closeAddModal() {
                      document.getElementById('addModal').classList.add('hidden');
                  }

                  function openEditModal(sales_data_id) {
                      currentSalesDataId = sales_data_id;
                      const row = rowsData.find(r => r.sales_data_id === sales_data_id);
                      if (row) {
                          ${fields
                              .filter(field => !['sales_data_id', 'day_of_week', 'Seasonal'].includes(field.name))
                              .map(field => `
                                  const ${field.name}Input = document.getElementById('edit_${field.name}');
                                  if (${field.name}Input) {
                                      ${field.name}Input.value = row['${field.name}'] || '';
                                  }
                              `).join('\n')}
                      }
                      document.getElementById('editModal').classList.remove('hidden');
                  }

                  function closeEditModal() {
                      document.getElementById('editModal').classList.add('hidden');
                  }
  
                  document.getElementById('addForm').addEventListener('submit', function(event) {
                      event.preventDefault();
                      
                      const data = {};
                      ${fields
                        .filter((field) => !['sales_data_id', 'day_of_week', 'Seasonal'].includes(field.name))
                        .map(
                          (field) => `
                          const ${field.name}Value = document.getElementById('add_${field.name}').value;
                          data['${field.name}'] = ${['sales_amount', 'profit_amount'].includes(field.name) ? 
                            `${field.name}Value === '' ? null : ${field.name}Value` : 
                            `${field.name}Value`};`
                        )
                        .join("\n")}
                      
                      axios.post('/Salesdata', data)
                          .then(response => {
                              alert('Record added successfully');
                              closeAddModal();
                              window.location.reload();
                          })
                          .catch(error => {
                              console.error('Error:', error);
                              alert('Failed to add the record');
                          });
                  });

                  document.getElementById('editForm').addEventListener('submit', function(event) {
                      event.preventDefault();
                      
                      const data = {};
                      ${fields
                        .filter((field) => !['sales_data_id', 'day_of_week', 'Seasonal'].includes(field.name))
                        .map(
                          (field) => `
                          const ${field.name}Value = document.getElementById('edit_${field.name}').value;
                          data['${field.name}'] = ${['sales_amount', 'profit_amount'].includes(field.name) ? 
                            `${field.name}Value === '' ? null : ${field.name}Value` : 
                            `${field.name}Value`};`
                        )
                        .join("\n")}
                      
                      axios.put(\`/Salesdata/\${currentSalesDataId}\`, data)
                          .then(response => {
                              alert('Record updated successfully');
                              closeEditModal();
                              window.location.reload();
                          })
                          .catch(error => {
                              console.error('Error:', error);
                              alert('Failed to update the record');
                          });
                  });
  
                  function deleteRow(sales_data_id) {
                      if (confirm('Are you sure you want to delete this record?')) {
                          axios.delete(\`/Salesdata/\${sales_data_id}\`)
                              .then(response => {
                                  alert('Record deleted successfully');
                                  window.location.reload();
                              })
                              .catch(error => {
                                  console.error('Error:', error);
                                  alert('Failed to delete the record');
                              });
                      }
                  }
              </script>
          </body>
          </html>
      `;
  }
  
  module.exports = generateHtmlPage;