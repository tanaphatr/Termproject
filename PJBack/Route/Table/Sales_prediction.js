// routes/dataOfProRoutes.js
const express = require('express');
const router = express.Router();
const db = require('../../connect.js');
const generateHtmlPage = require('../Tabletemplate.js');

// Route to fetch Sales_prediction
router.get('/html', async (req, res) => {
    try {
        const [rows, fields] = await db.query('SELECT * FROM Sales_prediction');
        res.send(generateHtmlPage('Data of Pro', fields, rows));
    } catch (err) {
        console.error('Error fetching Sales_prediction:', err);
        res.status(500).json({ error: 'Error fetching Sales_prediction' });
    }
});

router.get('/', async (req, res) => {
    try {
        const [rows, fields] = await db.query('SELECT * FROM Sales_prediction');
        res.json(rows);
    } catch (err) {
        console.error('Error fetching Sales_prediction:', err);
        res.status(500).json({ error: 'Error fetching Sales_prediction' });
    }
});

module.exports = router;