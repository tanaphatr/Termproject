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

router.get('/:user_id?', async (req, res) => {
    try {
        const { user_id } = req.params; // ดึง user_id จาก URL

        if (user_id) {
            // ถ้ามี user_id ให้ดึงข้อมูลเฉพาะ ID นั้น
            const [rows] = await db.query('SELECT * FROM Users WHERE user_id = ?', [user_id]);
            
            if (rows.length === 0) {
                return res.status(404).json({ error: "User not found" }); // ถ้าไม่พบผู้ใช้
            }

            return res.json(rows[0]); // ส่งข้อมูลเฉพาะ user
        } else {
            // ถ้าไม่มี user_id ให้ดึงข้อมูลทั้งหมด
            const [rows] = await db.query('SELECT * FROM Users');
            return res.json(rows); // ส่งข้อมูลทั้งหมด
        }
    } catch (err) {
        console.error('Error fetching Users:', err);
        res.status(500).json({ error: 'Error fetching Users' });
    }
});


router.post('/', async (req, res) => {
    const { username, password_hash, role } = req.body;

    // Validation
    if (!username || !password_hash || !role) {  // ตรวจสอบว่ามีการส่งค่าครบถ้วนหรือไม่
        return res.status(400).json({ error: "Missing required fields" });
    }

    try {
        const result = await db.query(
            'INSERT INTO Users (username, password_hash, role) VALUES (?, ?, ?)', 
            [username, password_hash, role]
        );
        
        res.status(201).json({ user_id: result.insertId, username, password_hash, role });
    } catch (err) {
        console.error('Error adding record:', err);
        res.status(500).json({ error: 'Error adding record' });
    }
});


// PUT: อัปเดตข้อมูล Users โดยอ้างอิงจาก user_id
router.put('/:user_id', async (req, res) => {
    const { user_id } = req.params;
    const { username, password_hash, role } = req.body;

    // Validation ตรวจสอบว่า field จำเป็นมีข้อมูลครบหรือไม่
    if (!username || !password_hash || !role) {
        return res.status(400).json({ error: "Missing required fields" });
    }

    try {
        // อัปเดตข้อมูลในฐานข้อมูลโดยใช้ user_id ที่ดึงมา
        const result = await db.query(
            'UPDATE Users SET username = ?, password_hash = ?, role = ? WHERE user_id = ?',
            [username, password_hash, role, user_id] // ใส่ user_id ตรงนี้
        );

        if (result.affectedRows === 0) {
            return res.status(404).json({ error: "Record not found" });
        }

        res.status(200).json({ message: "Record updated successfully", user_id, username, password_hash, role });
    } catch (err) {
        console.error('Error updating record:', err);
        res.status(500).json({ error: 'Error updating record' });
    }
});

// DELETE: ลบข้อมูล Users โดยอ้างอิงจาก user_id
router.delete('/:user_id', async (req, res) => {
    try {
        const { user_id } = req.params;

        if (!user_id) {
            return res.status(400).json({ error: "User ID is required" });
        }

        console.log("Deleting User ID:", user_id);

        const result = await db.query('DELETE FROM Users WHERE user_id = ?', [user_id]);

        if (result.affectedRows === 0) {
            return res.status(404).json({ error: "Users record not found" });
        }

        res.status(200).json({ message: "Users record deleted successfully" });
    } catch (err) {
        console.error('Error deleting record:', err);
        res.status(500).json({ error: err.message || "Internal server error" });
    }
});


module.exports = router;