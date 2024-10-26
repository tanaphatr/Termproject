// routes/dataOfProRoutes.js
const express = require('express');
const router = express.Router();
const db = require('../../connect.js');
const generateHtmlPage = require('../Tabletemplate.js');

// Route to fetch products in HTML format
router.get('/html', async (req, res) => {
    try {
        const [rows, fields] = await db.query('SELECT * FROM products');
        res.send(generateHtmlPage('Data of Pro', fields, rows));
    } catch (err) {
        console.error('Error fetching products:', err);
        res.status(500).json({ error: 'Error fetching products' });
    }
});

// Route to fetch products in JSON format
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

// Route to update an existing product
router.put('/:product_id', async (req, res) => {
    const { product_id } = req.params;
    console.log(product_id);
    const { product_code, name, category, stock_quantity, unit_price, min_stock_level } = req.body;
    try {
        const result = await db.query(
            'UPDATE products SET product_code = ?, name = ?, category = ?, stock_quantity = ?, unit_price = ?, min_stock_level = ? WHERE product_id = ?',
            [product_code, name, category, stock_quantity, unit_price, min_stock_level, product_id] // ใช้ product_id แทน id
        );

        if (result.affectedRows > 0) {
            res.json({ message: 'Product updated successfully' });
        } else {
            res.status(404).json({ error: 'Product not found' });
        }
    } catch (err) {
        console.error('Error updating product:', err);
        res.status(500).json({ error: 'Error updating product' });
    }
});


// Route to delete a product
router.delete('/:product_id', async (req, res) => {
    const { product_id } = req.params; // รับค่า product_id จาก params
    console.log(product_id);
    try {
        const result = await db.query('DELETE FROM products WHERE product_id = ?', [product_id]);
        
        // ตรวจสอบว่าได้ทำการลบข้อมูลหรือไม่
        if (result.affectedRows > 0) {
            res.json({ message: 'Product deleted successfully' });
        } else {
            res.status(404).json({ error: 'Product not found' }); // หากไม่พบผลิตภัณฑ์
        }
    } catch (err) {
        console.error('Error deleting product:', err);
        res.status(500).json({ error: 'Error deleting product' });
    }
});



module.exports = router;
