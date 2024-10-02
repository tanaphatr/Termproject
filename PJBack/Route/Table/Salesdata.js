// routes/dataOfProRoutes.js
const express = require('express');
const router = express.Router();
const db = require('../../connect.js');
const generateHtmlPage = require('../Tabletemplate.js');

// Route to fetch Salesdata
router.get('/html', async (req, res) => {
    try {
        const [rows, fields] = await db.query('SELECT * FROM Salesdata');
        res.send(generateHtmlPage('Data of Pro', fields, rows));
    } catch (err) {
        console.error('Error fetching Salesdata:', err);
        res.status(500).json({ error: 'Error fetching Salesdata' });
    }
});

router.get('/', async (req, res) => {
    try {
        const [rows, fields] = await db.query('SELECT * FROM Salesdata');
        res.json(rows);
    } catch (err) {
        console.error('Error fetching Salesdata:', err);
        res.status(500).json({ error: 'Error fetching Salesdata' });
    }
});

module.exports = router;