// routes/dataOfProRoutes.js
const express = require('express');
const router = express.Router();
const db = require('../../connect.js');
const generateHtmlPage = require('../Tabletemplate.js');

// Route to fetch Employees
router.get('/html', async (req, res) => {
    try {
        const [rows, fields] = await db.query('SELECT * FROM Employees');
        res.send(generateHtmlPage('Data of Pro', fields, rows));
    } catch (err) {
        console.error('Error fetching Employees:', err);
        res.status(500).json({ error: 'Error fetching Employees' });
    }
});

router.get('/', async (req, res) => {
    try {
        const [rows, fields] = await db.query('SELECT * FROM Employees');
        res.json(rows);
    } catch (err) {
        console.error('Error fetching Employees:', err);
        res.status(500).json({ error: 'Error fetching Employees' });
    }
});

module.exports = router;