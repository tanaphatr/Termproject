// routes/dataOfProRoutes.js
const express = require('express');
const router = express.Router();
const db = require('../../connect.js');
const generateHtmlPage = require('../Tabletemplate.js');

// Route to fetch Users
router.get('/html', async (req, res) => {
    try {
        const [rows, fields] = await db.query('SELECT * FROM Users');
        res.send(generateHtmlPage('Data of Pro', fields, rows));
    } catch (err) {
        console.error('Error fetching Users:', err);
        res.status(500).json({ error: 'Error fetching Users' });
    }
});

router.get('/', async (req, res) => {
    try {
        const [rows, fields] = await db.query('SELECT * FROM Users');
        res.json(rows);
    } catch (err) {
        console.error('Error fetching Users:', err);
        res.status(500).json({ error: 'Error fetching Users' });
    }
});

module.exports = router;