const express = require('express');
const router = express.Router();
const db = require('../../connect.js');
const generateHtmlPage = require('../Tabletemplate.js');

// Route to fetch daily sales data with employee details
router.get('/html', async (req, res) => {
    try {
        const [rows, fields] = await db.query(`
            SELECT daily_sales.*, employees.first_name, employees.last_name 
            FROM daily_sales 
            JOIN employees ON daily_sales.employee_id = employees.employee_id
        `);
        res.render('pages/Crudform', {
            title: 'Daily Sales Data',
            fields: fields,
            rows: rows,
            res: res,
            table_name: "Daily_sales",
            primary_key: "daily_sale_id"
        });
    } catch (err) {
        console.error('Error fetching daily sales:', err);
        res.status(500).json({ error: 'Error fetching daily sales' });
    }
});

// Route to fetch all daily sales data
router.get('/', async (req, res) => {
    try {
        const [rows] = await db.query('SELECT * FROM daily_sales');
        res.json(rows);
    } catch (err) {
        console.error('Error fetching daily sales:', err);
        res.status(500).json({ error: 'Error fetching daily sales' });
    }
});

// เพิ่มข้อมูลยอดขายรายวัน (POST)
router.post('/', async (req, res) => {
    const { sale_date, total_sales, employee_id } = req.body;

    if (!sale_date || !total_sales || !employee_id) {
        return res.status(400).json({ error: "Missing required fields" });
    }

    try {
        // ตรวจสอบว่า employee_id มีอยู่จริงในตาราง employees หรือไม่
        const [employee] = await db.query('SELECT * FROM employees WHERE employee_id = ?', [employee_id]);
        if (employee.length === 0) {
            return res.status(404).json({ error: "Employee not found" });
        }

        const result = await db.query(
            'INSERT INTO daily_sales (sale_date, total_sales, employee_id) VALUES (?, ?, ?)', 
            [sale_date, total_sales, employee_id]
        );

        res.status(201).json({ daily_sale_id: result.insertId, sale_date, total_sales, employee_id });
    } catch (err) {
        console.error('Error adding daily sales record:', err);
        res.status(500).json({ error: 'Error adding daily sales record' });
    }
});

// แก้ไขข้อมูลยอดขายรายวัน (PUT)
router.put('/:daily_sale_id', async (req, res) => {
    const { daily_sale_id } = req.params;
    const { sale_date, total_sales, employee_id } = req.body;

    if (!sale_date || !total_sales || !employee_id) {
        return res.status(400).json({ error: "Missing required fields" });
    }

    try {
        // ตรวจสอบว่า employee_id มีอยู่จริงในตาราง employees หรือไม่
        const [employee] = await db.query('SELECT * FROM employees WHERE employee_id = ?', [employee_id]);
        if (employee.length === 0) {
            return res.status(404).json({ error: "Employee not found" });
        }

        const result = await db.query(
            'UPDATE daily_sales SET sale_date = ?, total_sales = ?, employee_id = ? WHERE daily_sale_id = ?',
            [sale_date, total_sales, employee_id, daily_sale_id]
        );

        if (result.affectedRows === 0) {
            return res.status(404).json({ error: "Record not found" });
        }

        res.status(200).json({ message: "Record updated successfully", daily_sale_id, sale_date, total_sales, employee_id });
    } catch (err) {
        console.error('Error updating daily sales record:', err);
        res.status(500).json({ error: 'Error updating daily sales record', details: err.message });
    }
});

// Route to delete a daily sales record
router.delete('/:daily_sale_id', async (req, res) => {
    try {
        const { daily_sale_id } = req.params;

        const [result] = await db.query('DELETE FROM daily_sales WHERE daily_sale_id = ?', [daily_sale_id]);

        if (result.affectedRows === 0) {
            return res.status(404).json({ error: "Sales record not found" });
        }

        res.status(200).json({ message: "Sales record deleted successfully" });
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: err.message || "Internal server error" });
    }
});

module.exports = router;
