// routes/dataOfProRoutes.js
const express = require('express');
const router = express.Router();
const db = require('../../connect.js');
const generateHtmlPage = require('../Tabletemplate.js');

// Route to fetch Product_sales
router.get('/html', async (req, res) => {
    try {
        const [rows, fields] = await db.query('SELECT * FROM product_sales');
        res.render('pages/Crudform',{title:'Product Data', fields: fields,rows: rows,res:res, table_name:"Product_sales",primary_key: "product_sale_id"});
    } catch (err) {
        console.error('Error fetching product_sales:', err);
        res.status(500).json({ error: 'Error fetching product_sales' });
    }
});

router.get('/', async (req, res) => {
    try {
        const [rows, fields] = await db.query('SELECT * FROM product_sales');
        res.json(rows);
    } catch (err) {
        console.error('Error fetching product_sales:', err);
        res.status(500).json({ error: 'Error fetching product_sales' });
    }
});

module.exports = router;