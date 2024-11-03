const express = require('express');
const router = express.Router();
const db = require('../../connect.js');
const generateHtmlPage = require('../Tabletemplate.js');

// Route to fetch Salesdata and display as HTML
router.get('/html', async (req, res) => {
    try {
        const [rows, fields] = await db.query('SELECT * FROM Salesdata');
        res.send(generateHtmlPage('Data of Pro', fields, rows));
    } catch (err) {
        console.error('Error fetching Salesdata:', err);
        res.status(500).json({ error: 'Error fetching Salesdata' });
    }
});

// Route to fetch all Salesdata (JSON format)
router.get('/', async (req, res) => {
    try {
        const [rows] = await db.query('SELECT * FROM Salesdata');
        res.json(rows);
    } catch (err) {
        console.error('Error fetching Salesdata:', err);
        res.status(500).json({ error: 'Error fetching Salesdata' });
    }
});

// Route to add a new Salesdata record (POST)
router.post('/', async (req, res) => {
    const { sale_date, sales_amount, profit_amount, event, festival, weather, Temperature, Back_to_School_Period } = req.body;

    // Validation
    if (!sale_date) {
        return res.status(400).json({ error: "Missing required fields" });
    }

    try {
        const result = await db.query(
            'INSERT INTO Salesdata (sale_date, sales_amount, profit_amount, event, festival, weather, Temperature, Back_to_School_Period) VALUES (?, ?, ?, ?, ?, ?, ?, ?)', 
            [sale_date, sales_amount, profit_amount, event, festival, weather, Temperature, Back_to_School_Period]
        );
        
        res.status(201).json({ id: result.insertId, sale_date, sales_amount, profit_amount, event, festival, weather, Temperature, Back_to_School_Period});
    } catch (err) {
        console.error('Error adding record:', err);
        res.status(500).json({ error: 'Error adding record' });
    }
});

router.put('/:sales_data_id', async (req, res) => {
    const { sales_data_id } = req.params; // ดึง sales_data_id จาก URL
    const { sales_amount, profit_amount, event, day_of_week, festival, weather, Temperature, Back_to_School_Period, Seasonal } = req.body;

    // Validation ตรวจสอบว่า field จำเป็นมีข้อมูลครบหรือไม่
    if (!sales_data_id || !sales_amount || !profit_amount) {
        return res.status(400).json({ error: "Missing required fields" });
    }

    try {
        // อัปเดตข้อมูลในฐานข้อมูลโดยใช้ sales_data_id ที่ดึงมา
        const result = await db.query(
            'UPDATE Salesdata SET  sales_amount = ?, profit_amount = ?, event = ?, day_of_week = ?, festival = ?, weather = ?, Temperature = ?, Back_to_School_Period = ?, Seasonal = ? WHERE sales_data_id = ?',
            [ sales_amount, profit_amount, event, day_of_week, festival, weather, Temperature, Back_to_School_Period, Seasonal, sales_data_id]
        );

        if (result.affectedRows === 0) {
            return res.status(404).json({ error: "Record not found" });
        }

        res.status(200).json({ message: "Record updated successfully", sales_data_id,  sales_amount, profit_amount, event, day_of_week, festival, weather, Temperature, Back_to_School_Period, Seasonal });
    } catch (err) {
        console.error('Error updating record:', err);
        res.status(500).json({ error: 'Error updating record' });
    }
});

router.delete('/:sales_data_id', async (req, res) => {
    try {
        const { sales_data_id } = req.params;
        if (!sales_data_id) {
            return res.status(400).json({ error: "Sales data ID is required" });
        }

        console.log("Deleting Salesdata ID:", sales_data_id);

        const [result] = await db.query('DELETE FROM Salesdata WHERE sales_data_id = ?', [sales_data_id]);

        if (result.affectedRows === 0) {
            return res.status(404).json({ error: "Salesdata record not found" });
        }

        res.status(200).json({ message: "Salesdata record deleted successfully" });
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: err.message || "Internal server error" });
    }
});

module.exports = router;

