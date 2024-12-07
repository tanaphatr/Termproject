// routes/dataOfProRoutes.js
const express = require('express');
const router = express.Router();
const db = require('../../connect.js');

// GET - Render HTML page with product sales data
router.get('/html', async (req, res) => {
    try {
        const [rows, fields] = await db.query('SELECT * FROM product_sales');
        res.render('pages/Crudform', {
            title: 'Product Data',
            fields: fields, 
            rows: rows,
            res: res,
            table_name: "product_sales",
            primary_key: "product_sale_id"
        });
    } catch (err) {
        console.error('Error fetching product_sales:', err);
        res.status(500).json({ error: 'Error fetching product_sales' });
    }
});

// GET - Fetch all product sales (API)
router.get('/', async (req, res) => {
    try {
        const [rows] = await db.query('SELECT * FROM product_sales');
        res.json(rows);
    } catch (err) {
        console.error('Error fetching product_sales:', err);
        res.status(500).json({ error: 'Error fetching product_sales' });
    }
});

// GET - Fetch single product sale by ID
router.get('/:id', async (req, res) => {
    try {
        const [rows] = await db.query('SELECT * FROM product_sales WHERE product_sale_id = ?', [req.params.id]);
        if (rows.length === 0) {
            return res.status(404).json({ error: 'Product sale not found' });
        }
        res.json(rows[0]);
    } catch (err) {
        console.error('Error fetching product sale:', err);
        res.status(500).json({ error: 'Error fetching product sale' });
    }
});

// POST - Create new product sale
router.post('/', async (req, res) => {
    try {
        const {Product_code, Date, Quantity, Total_Sale} = req.body;
        const [result] = await db.query(
            'INSERT INTO product_sales (Product_code, Date, Quantity, Total_Sale) VALUES (?, ?, ?, ?)',
            [Product_code, Date, Quantity, Total_Sale]
        );
        
        res.status(201).json({
            message: 'Product sale created successfully',
            product_sale_id: result.insertId
        });
    } catch (err) {
        console.error('Error creating product sale:', err);
        res.status(500).json({ error: 'Error creating product sale' });
    }
});

// PUT - Update product sale
router.put('/:id', async (req, res) => {
    try {
        const {product_code, Date, Quantity, Total_Sale} = req.body;
        const [result] = await db.query(
            'UPDATE product_sales SET product_code = ?, Date = ?, Quantity = ?, Total_Sale = ? WHERE product_sale_id = ?',
            [product_code, Date, Quantity, Total_Sale, req.params.id]
        );
        
        if (result.affectedRows === 0) {
            return res.status(404).json({ error: 'Product sale not found' });
        }
        
        res.json({ message: 'Product sale updated successfully' });
    } catch (err) {
        console.error('Error updating product sale:', err);
        res.status(500).json({ error: 'Error updating product sale' });
    }
});

// DELETE - Delete product sale
router.delete('/:id', async (req, res) => {
    try {
        const [result] = await db.query('DELETE FROM product_sales WHERE product_sale_id = ?', [req.params.id]);
        
        if (result.affectedRows === 0) {
            return res.status(404).json({ error: 'Product sale not found' });
        }
        
        res.json({ message: 'Product sale deleted successfully' });
    } catch (err) {
        console.error('Error deleting product sale:', err);
        res.status(500).json({ error: 'Error deleting product sale' });
    }
});

module.exports = router;