// routes/dataOfProRoutes.js
const express = require('express');
const router = express.Router();
const db = require('../../connect.js');
const generateHtmlPage = require('../Tabletemplate.js');

// Route to fetch products
router.get('/html', async (req, res) => {
    try {
        const [rows, fields] = await db.query('SELECT * FROM employees');
        // res.send(generateHtmlPage('Data of Pro', fields, rows));
        res.render('pages/Crudform',{title:'Employee Data', fields: fields,rows: rows,res:res, table_name:"Employees",primary_key: "employee_id"});
    } catch (err) {
        console.error('Error fetching employee:', err);
        res.status(500).json({ error: 'Error fetching employee' });
    }
});

router.get('/', async (req, res) => {
    try {
        const [rows, fields] = await db.query('SELECT * FROM employees');
        res.json(rows);
    } catch (err) {
        console.error('Error fetching employee:', err);
        res.status(500).json({ error: 'Error fetching employee' });
    }
});

router.post('/', async (req, res) => {
     const { employee_id, first_name, last_name, position, email,phone, nickname, age, address, district, province, salary } = req.body;
    
    // Validation
    if (!first_name || !last_name || !phone) {  // ตรวจสอบว่ามีการส่งค่าครบถ้วนหรือไม่
        return res.status(400).json({ error: "Missing required fields" });
    }

    try {
        const result = await db.query(
            'INSERT INTO employees (first_name, last_name, position, email,phone, nickname, age, address, district, province, salary) VALUES (?,?,?,?,?,?,?,?,?,?,?)', 
          [ first_name, last_name, position, email,phone, nickname, age, address, district, province, salary]
        );
        
        res.status(201).json({ employee_id: result.insertId, first_name, last_name, position, email,phone, nickname, age, address, district, province, salary });
    } catch (err) {
        console.error('Error adding record:', err);
        res.status(500).json({ error: 'Error adding record' });
    }
});


router.put('/:employee_id', async (req, res) => {
    const { employee_id } = req.params; // ดึง employee_id จาก URL
    const { first_name, last_name, position, email, phone, nickname, age, address, district, province, salary } = req.body;

    // Validation ตรวจสอบว่า field จำเป็นมีข้อมูลครบหรือไม่
    if (!first_name || !last_name || !phone) {
        return res.status(400).json({ error: "Missing required fields" });
    }

    // ตรวจสอบว่า employee_id มีค่าหรือไม่
    if (!employee_id) {
        return res.status(400).json({ error: "Employee ID is required" });
    }

    try {
        // อัปเดตข้อมูลในฐานข้อมูลโดยใช้ employee_id ที่ดึงมา
        const result = await db.query(
            'UPDATE employees SET first_name = ?, last_name = ?, position = ?, email = ?, phone = ?, nickname = ?, age = ?, address = ?, district = ?, province = ?, salary = ? WHERE employee_id = ?',
            [first_name, last_name, position, email, phone, nickname, age, address, district, province, salary, employee_id]
        );

        if (result.affectedRows === 0) {
            return res.status(404).json({ error: "Record not found" });
        }

        res.status(200).json({ message: "Record updated successfully", employee_id, first_name, last_name, position, email, phone, nickname, age, address, district, province, salary });
    } catch (err) {
        console.error('Error updating record:', err);
        res.status(500).json({ error: 'Error updating record', details: err.message });
    }
});



router.delete('/:employee_id', async (req, res) => {
    try {
        const { employee_id } = req.params;
        if (!employee_id) {
            return res.status(400).json({ error: "Product ID is required" });
        }

        console.log("Deleting Product ID:", employee_id);

        const [result] = await db.query('DELETE FROM employees WHERE employee_id = ?', [employee_id]);

        if (result.affectedRows === 0) {
            return res.status(404).json({ error: "Products  record not found" });
        }

        res.status(200).json({ message: "Products record deleted successfully" });
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: err.message || "Internal server error" });
    }
});

module.exports = router;