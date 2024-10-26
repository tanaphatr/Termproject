// routes/dataOfProRoutes.js
const express = require('express');
const router = express.Router();
const db = require('../../connect.js');
const generateHtmlPage = require('../Tabletemplate.js');

// Route to fetch products
router.get('/html', async (req, res) => {
    try {
        const [rows, fields] = await db.query('SELECT * FROM products');
        // res.send(generateHtmlPage('Data of Pro', fields, rows));
        res.render('pages/Crudform',{title:'Product Data', fields: fields,rows: rows,res:res, table_name:"Products",primary_key: "product_id"});
    } catch (err) {
        console.error('Error fetching products:', err);
        res.status(500).json({ error: 'Error fetching products' });
    }
});

router.get('/', async (req, res) => {
    try {
        const [rows, fields] = await db.query('SELECT * FROM products');
        res.json(rows);
    } catch (err) {
        console.error('Error fetching products:', err);
        res.status(500).json({ error: 'Error fetching products' });
    }
});

// Route to add a new product
router.post('/', async (req, res) => {
    const { product_code, name, category, stock_quantity, unit_price, min_stock_level } = req.body;
    try {
        const result = await db.query(
            'INSERT INTO products (product_code, name, category, stock_quantity, unit_price, min_stock_level) VALUES (?, ?, ?, ?, ?, ?)',
            [product_code, name, category, stock_quantity, unit_price, min_stock_level]
        );
        res.status(201).json({ message: 'Product added successfully', productId: result.insertId });
    } catch (err) {
        console.error('Error adding product:', err);
        res.status(500).json({ error: 'Error adding product' });
    }
});


router.put('/:product_id', async (req, res) => {
    const { product_id } = req.params; // ดึง sales_data_id จาก URL
    const { product_code, name, category, stock_quantity, unit_price, min_stock_level } = req.body;

    // Validation ตรวจสอบว่า field จำเป็นมีข้อมูลครบหรือไม่
    if (!product_code || !name || !category) {
        return res.status(400).json({ error: "Missing required fields" });
    }

    try {
        // อัปเดตข้อมูลในฐานข้อมูลโดยใช้ sales_data_id ที่ดึงมา
        const result = await db.query(
            'UPDATE products SET  product_code = ?, name = ?, category = ?, stock_quantity = ?, unit_price = ?, min_stock_level = ? WHERE product_id = ?',
            [ product_code, name, category, stock_quantity, unit_price, min_stock_level, product_id]
        );

        if (result.affectedRows === 0) {
            return res.status(404).json({ error: "Record not found" });
        }

        res.status(200).json({ message: "Record updated successfully", product_id,  product_code, name, category,stock_quantity, unit_price, min_stock_level });
    } catch (err) {
        console.error('Error updating record:', err);
        res.status(500).json({ error: 'Error updating record' });
    }
});


router.delete('/:product_id', async (req, res) => {
    try {
        const { product_id } = req.params;
        if (!product_id) {
            return res.status(400).json({ error: "Product ID is required" });
        }

        console.log("Deleting Product ID:", product_id);

        const [result] = await db.query('DELETE FROM products WHERE product_id = ?', [product_id]);

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