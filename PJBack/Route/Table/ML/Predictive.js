// routes/externalDataRoutes.js
const express = require('express');
const router = express.Router();
const axios = require('axios');

// Route to fetch data from external URL
router.get('/', async (req, res) => {
    try {
        const response = await axios.get('http://localhost:8887/');
        res.json(response.data); // ส่งคืนข้อมูลที่ได้รับจาก URL
    } catch (err) {
        console.error('Error fetching external data:', err);
        res.status(500).json({ error: 'Error fetching external data' });
    }
});

module.exports = router;
