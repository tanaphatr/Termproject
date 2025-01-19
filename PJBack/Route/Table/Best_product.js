const express = require('express');
const router = express.Router();
const db = require('../../connect.js');

router.get('/', async (req, res) => {
    try {
        const query = `
            SELECT 
                p.Product_code,
                SUM(ps.Quantity) AS total_quantity,  -- จำนวนสินค้าที่ขายทั้งหมด
                SUM(ps.Total_Sale) AS total_sales,   -- ยอดขายทั้งหมด
                SUM(ps.Total_Sale) / COUNT(DISTINCT YEAR(ps.date) * 100 + MONTH(ps.date)) AS avg_monthly_sales,  -- ค่าเฉลี่ยยอดขายรายเดือน
                SUM(ps.Quantity) / COUNT(DISTINCT YEAR(ps.date) * 100 + MONTH(ps.date)) AS avg_quantity_per_month  -- ค่าเฉลี่ยสินค้าที่ขายได้ต่อเดือน
            FROM Product_sales ps
            JOIN Products p ON ps.Product_code = p.Product_code
            GROUP BY p.Product_code
            ORDER BY total_quantity DESC
            LIMIT 2`;

        const [rows] = await db.query(query);

        if (!rows || rows.length === 0) {
            return res.status(404).json({ message: 'ไม่พบข้อมูลสินค้า' });
        }

        res.render('pages/Crudform', {
            title: 'Best Selling Data',
            fields: [
                { name: "Product_code", label: "รหัสสินค้า" },
                { name: "total_quantity", label: "จำนวนขายรวม (ชิ้น)" },
                { name: "total_sales", label: "ยอดขายรวม (บาท)" },
                { name: "avg_monthly_sales", label: "ค่าเฉลี่ยยอดขายรายเดือน (บาท)" },
                { name: "avg_quantity_per_month", label: "ค่าเฉลี่ยสินค้าที่ขายได้ต่อเดือน (ชิ้น)" }
            ],
            rows: rows.map(row => ({
                ...row,
                total_sales: row.total_sales.toLocaleString('th-TH', {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2
                }),
                avg_monthly_sales: (row.avg_monthly_sales || 0).toLocaleString('th-TH', {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2
                }),
                avg_quantity_per_month: (row.avg_quantity_per_month || 0).toLocaleString('th-TH', {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2
                })
            })),
            table_name: "Product_sales",
            primary_key: "Product_code"
        });
    } catch (err) {
        console.error('Error:', err);
        res.status(500).json({ 
            error: 'เกิดข้อผิดพลาดในการดึงข้อมูล', 
            details: err.message 
        });
    }
});

// API endpoint สำหรับดึงข้อมูลแบบ JSON
router.get('/api', async (req, res) => {
    try {
        const query = `
            SELECT 
                p.Product_code,
                SUM(ps.Quantity) AS total_quantity,  -- จำนวนสินค้าที่ขายทั้งหมด
                SUM(ps.Total_Sale) AS total_sales,   -- ยอดขายทั้งหมด
                SUM(ps.Total_Sale) / COUNT(DISTINCT YEAR(ps.date) * 100 + MONTH(ps.date)) AS avg_monthly_sales,  -- ค่าเฉลี่ยยอดขายรายเดือน
                SUM(ps.Quantity) / COUNT(DISTINCT YEAR(ps.date) * 100 + MONTH(ps.date)) AS avg_quantity_per_month  -- ค่าเฉลี่ยสินค้าที่ขายได้ต่อเดือน
            FROM Product_sales ps
            JOIN Products p ON ps.Product_code = p.Product_code
            WHERE p.Product_code LIKE 'A%'  -- เพิ่มเงื่อนไขเพื่อเลือกเฉพาะสินค้าที่รหัสเริ่มต้นด้วย A
            GROUP BY p.Product_code
            ORDER BY total_quantity DESC
            LIMIT 2`;

        const [rows] = await db.query(query);

        if (!rows || rows.length === 0) {
            return res.status(404).json({ message: 'ไม่พบข้อมูลสินค้า' });
        }

        res.json(rows);
    } catch (err) {
        console.error('Error:', err);
        res.status(500).json({ 
            error: 'เกิดข้อผิดพลาดในการดึงข้อมูล', 
            details: err.message 
        });
    }
});

module.exports = router;
