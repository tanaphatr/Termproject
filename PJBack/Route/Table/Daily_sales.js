// routes/dataOfProRoutes.js
const express = require('express');
const router = express.Router();
const db = require('../../connect.js');
const generateHtmlPage = require('../Tabletemplate.js');

// Route to fetch daily_sales
router.get('/html', async (req, res) => {
    try {
        const [rows, fields] = await db.query('SELECT * FROM daily_sales');
        res.send(generateHtmlPage('Data of Pro', fields, rows));
    } catch (err) {
        console.error('Error fetching daily_sales:', err);
        res.status(500).json({ error: 'Error fetching daily_sales' });
    }
});

router.get('/', async (req, res) => {
    try {
        const [rows, fields] = await db.query('SELECT * FROM daily_sales');
        res.json(rows);
    } catch (err) {
        console.error('Error fetching daily_sales:', err);
        res.status(500).json({ error: 'Error fetching daily_sales' });
    }
});

module.exports = router;