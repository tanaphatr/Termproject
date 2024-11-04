const express = require('express');
const router = express.Router();
const db = require('../../connect.js');
const generateHtmlPage = require('../Tabletemplate.js');

// Route to fetch Product_stock_history in HTML format
router.get('/html', async (req, res) => {
    try {
        const [rows, fields] = await db.query('SELECT * FROM Product_stock_history');
        res.render('pages/Crudform', {
            title: 'Product Stock History Data',
            fields: fields,
            rows: rows,
            res: res,
            table_name: "Product_stock_history",
            primary_key: "stock_history_id"
        });
    } catch (err) {
        console.error('Error fetching Product_stock_history:', err);
        res.status(500).json({ error: 'Error fetching Product_stock_history' });
    }
});

// Route to fetch Product_stock_history in JSON format
router.get('/', async (req, res) => {
    try {
        const [rows, fields] = await db.query('SELECT * FROM Product_stock_history');
        res.json(rows);
    } catch (err) {
        console.error('Error fetching Product_stock_history:', err);
        res.status(500).json({ error: 'Error fetching Product_stock_history' });
    }
});

// Route to add a new stock history record
router.post('/', async (req, res) => {
    const { product_id, adjustment_date, adjustment_type, adjustment_quantity, adjusted_by } = req.body;
    try {
        const result = await db.query(
            'INSERT INTO Product_stock_history (product_id, adjustment_date, adjustment_type, adjustment_quantity, adjusted_by) VALUES (?, ?, ?, ?, ?)',
            [product_id, adjustment_date, adjustment_type, adjustment_quantity, adjusted_by]
        );
        res.status(201).json({ 
            message: 'Stock history record added successfully', 
            stockHistoryId: result.insertId 
        });
    } catch (err) {
        console.error('Error adding stock history record:', err);
        res.status(500).json({ error: 'Error adding stock history record' });
    }
});

// Route to update a stock history record
router.put('/:stock_history_id', async (req, res) => {
    const { stock_history_id } = req.params;
    const { product_id, adjustment_date, adjustment_type, adjustment_quantity, adjusted_by } = req.body;

    // Validation
    if (!product_id || !adjustment_date || !adjustment_type || !adjustment_quantity || !adjusted_by) {
        return res.status(400).json({ error: "Missing required fields" });
    }

    try {
        const result = await db.query(
            'UPDATE Product_stock_history SET product_id = ?, adjustment_date = ?, adjustment_type = ?, adjustment_quantity = ?, adjusted_by = ? WHERE stock_history_id = ?',
            [product_id, adjustment_date, adjustment_type, adjustment_quantity, adjusted_by, stock_history_id]
        );

        if (result.affectedRows === 0) {
            return res.status(404).json({ error: "Record not found" });
        }

        res.status(200).json({ 
            message: "Record updated successfully", 
            stock_history_id,
            product_id, 
            adjustment_date, 
            adjustment_type, 
            adjustment_quantity, 
            adjusted_by 
        });
    } catch (err) {
        console.error('Error updating record:', err);
        res.status(500).json({ error: 'Error updating record' });
    }
});

// Route to delete a stock history record
router.delete('/:stock_history_id', async (req, res) => {
    try {
        const { stock_history_id } = req.params;
        if (!stock_history_id) {
            return res.status(400).json({ error: "Stock History ID is required" });
        }

        console.log("Deleting Stock History ID:", stock_history_id);

        const [result] = await db.query(
            'DELETE FROM Product_stock_history WHERE stock_history_id = ?', 
            [stock_history_id]
        );

        if (result.affectedRows === 0) {
            return res.status(404).json({ error: "Stock history record not found" });
        }

        res.status(200).json({ message: "Stock history record deleted successfully" });
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: err.message || "Internal server error" });
    }
});

module.exports = router;