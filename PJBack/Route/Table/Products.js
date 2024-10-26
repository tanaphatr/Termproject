// routes/dataOfProRoutes.js
const express = require('express');
const router = express.Router();
const db = require('../../connect.js');
const generateHtmlPage = require('../Tabletemplate.js');

// Route to fetch products in HTML format
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

module.exports = router;
