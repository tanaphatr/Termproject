// routes/dataOfProRoutes.js
const express = require('express');
const router = express.Router();
const db = require('../../connect.js');
const generateHtmlPage = require('../Tabletemplate.js');

// Route to fetch Product_stock_history
router.get('/html', async (req, res) => {
    try {
        const [rows, fields] = await db.query('SELECT * FROM Product_stock_history');
        res.send(generateHtmlPage('Data of Pro', fields, rows));
    } catch (err) {
        console.error('Error fetching Product_stock_history:', err);
        res.status(500).json({ error: 'Error fetching Product_stock_history' });
    }
});

router.get('/', async (req, res) => {
    try {
        const [rows, fields] = await db.query('SELECT * FROM Product_stock_history');
        res.json(rows);
    } catch (err) {
        console.error('Error fetching Product_stock_history:', err);
        res.status(500).json({ error: 'Error fetching Product_stock_history' });
    }
});

module.exports = router;