const mysql = require('mysql2');

//Use for login Mysql 

// const config = {
//     user: 'Admin',
//     password: 'CE498termprojectsql',
//     host: 'localhost',
//     database: 'termproject',
//     waitForConnections: true,
//     connectionLimit: 10,
//     queueLimit: 0
// };

const config = {
    user: 'root',
    password: '',
    host: 'localhost',
    database: 'termproject',
    waitForConnections: true,
    connectionLimit: 10,
    queueLimit: 0
};

//Connect
const pool = mysql.createPool(config);

// Testing the connection
pool.getConnection((err, connection) => {
    if (err) {
        console.error('Error connecting to database:', err.stack);
        return;
    }
    console.log('Connected to the database');
    connection.release(); // release the connection
});

module.exports = pool.promise(); // Export the pool for queries