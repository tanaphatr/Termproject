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
    const { sale_date, sales_amount, profit_amount, event, day_of_week, festival, weather, Temperature, Back_to_School_Period, Seasonal } = req.body;

    try {
        const result = await db.query(
            'INSERT INTO Salesdata (sale_date, sales_amount, profit_amount, event, day_of_week, festival, weather, Temperature, Back_to_School_Period, Seasonal) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', 
            [sale_date, sales_amount, profit_amount, event, day_of_week, festival, weather, Temperature, Back_to_School_Period, Seasonal]
        );
        
        res.status(201).json({ id: result.insertId, sale_date, sales_amount, profit_amount, event, day_of_week, festival, weather, Temperature, Back_to_School_Period, Seasonal });
    } catch (err) {
        console.error('Error adding record:', err);
        res.status(500).json({ error: 'Error adding record' });
    }
});

// Route to update a Salesdata record (PUT)
router.put('/:id', async (req, res) => {
    const { id } = req.params;
    const { sales_data_id,sale_date, sales_amount, profit_amount, event, day_of_week, festival, weather, Temperature, Back_to_School_Period, Seasonal } = req.body;

    try {
        await db.query(
            'UPDATE Salesdata SET sales_data_id = ?, sale_date = ?, sales_amount = ?, profit_amount = ?, event = ?, day_of_week = ?, festival = ?, weather = ?, Temperature = ?, Back_to_School_Period = ?, Seasonal = ? WHERE sales_data_id = ?', 
            [sales_data_id,sale_date, sales_amount, profit_amount, event, day_of_week, festival, weather, Temperature, Back_to_School_Period, Seasonal, sales_data_id]
        );
        res.status(200).json({ message: 'Record updated successfully' });
    } catch (err) {
        console.error('Error updating record:', err);
        res.status(500).json({ error: 'Error updating record' });
    }
});

// Route to delete a Salesdata record (DELETE)
router.delete('/:sales_data_id', async (req, res) => {
    const { sales_data_id } = req.params;
    try {
        await db.query('DELETE FROM Salesdata WHERE sales_data_id = ?', [sales_data_id]);
        res.status(200).json({ message: 'Record deleted successfully' });
    } catch (err) {
        console.error('Error deleting record:', err);
        res.status(500).json({ error: 'Error deleting record' });
    }
});

module.exports = router;
