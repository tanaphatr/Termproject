// routes/salesPredictionRoutes.js
const express = require('express');
const router = express.Router();
const db = require('../../connect.js');

// GET - Render HTML page
router.get('/html', async (req, res) => {
    try {
        const [rows, fields] = await db.query('SELECT * FROM sales_prediction');
        res.render('pages/Crudform', {
            title: 'Prediction Data',
            fields: fields,
            rows: rows,
            res: res,
            table_name: "Sales_prediction",
            primary_key: "prediction_id"
        });
    } catch (err) {
        console.error('Error fetching sales predictions:', err);
        res.status(500).json({ error: 'Error fetching sales predictions' });
    }
});

// GET - Fetch all predictions
router.get('/', async (req, res) => {
    try {
        const [rows] = await db.query('SELECT * FROM sales_prediction ORDER BY prediction_date DESC');
        res.json(rows);
    } catch (err) {
        console.error('Error fetching predictions:', err);
        res.status(500).json({ error: 'Error fetching predictions' });
    }
});

// GET - Fetch single prediction
router.get('/:id', async (req, res) => {
    try {
        const [rows] = await db.query('SELECT * FROM sales_prediction WHERE prediction_id = ?', [req.params.id]);
        if (rows.length === 0) {
            return res.status(404).json({ error: 'Prediction not found' });
        }
        res.json(rows[0]);
    } catch (err) {
        console.error('Error fetching prediction:', err);
        res.status(500).json({ error: 'Error fetching prediction' });
    }
});

// POST - Create new prediction
router.post('/', async (req, res) => {
    try {
        const { prediction_date, predicted_sales} = req.body;
        const [result] = await db.query(
            'INSERT INTO sales_prediction (prediction_date, predicted_sales) VALUES (?, ?)',
            [prediction_date, predicted_sales]
        );
        
        res.status(201).json({
            message: 'Sales prediction created successfully',
            prediction_id: result.insertId
        });
    } catch (err) {
        console.error('Error creating prediction:', err);
        res.status(500).json({ error: 'Error creating prediction' });
    }
});

// PUT - Update prediction
router.put('/:id', async (req, res) => {
    try {
        const { prediction_date, predicted_sales, actual_sales, error_value } = req.body;
        const [result] = await db.query(
            'UPDATE sales_prediction SET prediction_date = ?, predicted_sales = ?, actual_sales = ?, error_value = ? WHERE prediction_id = ?',
            [prediction_date, predicted_sales, actual_sales, error_value, req.params.id]
        );
        
        if (result.affectedRows === 0) {
            return res.status(404).json({ error: 'Prediction not found' });
        }
        
        res.json({ message: 'Sales prediction updated successfully' });
    } catch (err) {
        console.error('Error updating prediction:', err);
        res.status(500).json({ error: 'Error updating prediction' });
    }
});

// DELETE - Delete prediction
router.delete('/:id', async (req, res) => {
    try {
        const [result] = await db.query('DELETE FROM sales_prediction WHERE prediction_id = ?', [req.params.id]);
        
        if (result.affectedRows === 0) {
            return res.status(404).json({ error: 'Prediction not found' });
        }
        
        res.json({ message: 'Sales prediction deleted successfully' });
    } catch (err) {
        console.error('Error deleting prediction:', err);
        res.status(500).json({ error: 'Error deleting prediction' });
    }
});

module.exports = router;