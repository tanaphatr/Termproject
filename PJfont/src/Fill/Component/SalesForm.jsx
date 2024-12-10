import React, { useEffect, useState } from 'react';
import {
    Button,
    Card,
    CardContent,
    TextField,
    Typography,
    MenuItem,
    Select,
    InputLabel,
    FormControl,
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Paper,
} from '@mui/material';

const SalesForm = () => {
    const [products, setProducts] = useState([]);
    const [open, setOpen] = useState(false);
    const [selectedProduct, setSelectedProduct] = useState('');
    const [quantity, setQuantity] = useState('');
    const [addedProducts, setAddedProducts] = useState([]);

    useEffect(() => {
        // ดึงข้อมูลจาก localStorage เมื่อมีการรีเฟรชหรือเปิดหน้าใหม่
        const storedProducts = JSON.parse(localStorage.getItem('addedProducts')) || [];
        setAddedProducts(storedProducts);
        const fetchProducts = async () => {
            try {
                const response = await fetch("http://localhost:8888/Products");
                if (!response.ok) {
                    throw new Error("Network response was not ok");
                }
                const data = await response.json();
                setProducts(data);
            } catch (error) {
                console.error("Error fetching Products:", error);
            }
        };

        fetchProducts();
    }, []);

    useEffect(() => {
        // เก็บข้อมูลลงใน localStorage ทุกครั้งที่ addedProducts เปลี่ยนแปลง
        if (addedProducts.length > 0) {
            localStorage.setItem('addedProducts', JSON.stringify(addedProducts));
        }

    }, [addedProducts]);
    
    const handleClickOpen = () => {
        setOpen(true);
    };

    const handleClose = () => {
        setOpen(false);
        setSelectedProduct('');
        setQuantity('');
    };

    const handleAddProduct = () => {
        const product = products.find(prod => prod.name === selectedProduct);
        const qty = parseInt(quantity);
    
        if (product && qty > 0) {  // Validate quantity
            const newProduct = {
                product_id: product.product_id,
                name: product.name,
                quantity: qty,
                unit_price: product.unit_price,
                total: product.unit_price * qty,
            };
            setAddedProducts(prevProducts => {
                const updatedProducts = [...prevProducts, newProduct];
                console.log('Updated addedProducts:', updatedProducts);  // ตรวจสอบค่าของ addedProducts หลังจากเพิ่มสินค้า
                return updatedProducts;
            });
            handleClose();
        } else {
            alert("Please select a product and enter a valid quantity.");
        }
    };
    

    const handleRemoveProduct = (product_id) => {
        setAddedProducts(prevProducts => {
            const updatedProducts = prevProducts.filter(prod => prod.product_id !== product_id);
    
            // อัพเดตข้อมูลลงใน localStorage หลังจากที่ลบสินค้า
            localStorage.setItem('addedProducts', JSON.stringify(updatedProducts));
    
            return updatedProducts;
        });
    };

    const handleReset = () => {
        setAddedProducts([]); // Clear added products
        localStorage.removeItem('addedProducts'); // ลบข้อมูลจาก localStorage
        setSelectedProduct('');
        setQuantity('');
    };

    const handleSave = async () => {
        // ดึง employee_id จาก localStorage
        const employee = JSON.parse(localStorage.getItem("loggedInUser"));
        const employeeId = employee ? employee.user_id : null;
        if (!employeeId) {
            alert("Employee ID not found in localStorage.");
            return;
        }

        const today = new Date().toISOString();
        // คำนวณยอดขายรวม
        const totalSale = addedProducts.reduce((total, product) => total + product.total, 0);

        // ส่งข้อมูลยอดขายรายวัน
        try {
            const dailySaleResponse = await fetch("http://localhost:8888/Daily_sales", {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    sale_date: today,
                    total_sales: totalSale,
                    employee_id: employeeId,
                }),
            });

            if (!dailySaleResponse.ok) {
                throw new Error("Failed to save daily sales.");
            }


            // ส่งข้อมูลยอดขายของแต่ละผลิตภัณฑ์
            for (const product of addedProducts) {
                const productSaleResponse = await fetch("http://localhost:8888/Product_sales", {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        product_id: product.product_id,
                        date: today,
                        quantity_sold: product.quantity,
                        sale_amount: product.total,
                    }),
                });

                if (!productSaleResponse.ok) {
                    throw new Error(`Failed to save product sale for ${product.name}.`);
                }
            }

            alert("Sales data saved successfully!");

            // รีเซ็ตข้อมูลหลังจากการบันทึก
            handleReset();
        } catch (error) {
            console.error("Error saving sales data:", error);
            alert("There was an error saving the sales data.");
        }
    };

    return (
        <Card style={{ padding: '20px' }}>
            <CardContent>
                <Typography variant="h5" align="center">Sales Form</Typography>
                <TableContainer component={Paper} style={{ marginTop: '20px' }}>
                    <Table>
                        <TableHead>
                            <TableRow>
                                <TableCell align="center">Nameproduct</TableCell>
                                <TableCell align="center">Quantity</TableCell>
                                <TableCell align="center">Unitprice</TableCell>
                                <TableCell align="center">Total</TableCell>
                                <TableCell align="center">Action</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {addedProducts.map((product) => (
                                <TableRow key={product.product_id}>
                                    <TableCell align="center">{product.name}</TableCell>
                                    <TableCell align="center">{product.quantity}</TableCell>
                                    <TableCell align="center">{product.unit_price}</TableCell>
                                    <TableCell align="center">{product.total}</TableCell>
                                    <TableCell align="center">
                                        <Button
                                            variant="outlined"
                                            color="error"
                                            onClick={() => handleRemoveProduct(product.product_id)}
                                        >
                                            Delete
                                        </Button>
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </TableContainer>
                <div style={{ display: 'flex', justifyContent: 'center', marginTop: '20px', gap: '10px' }}>
                    <Button variant="outlined" color="primary" onClick={handleClickOpen}>
                        Add Product
                    </Button>
                    <Button variant="outlined" onClick={handleReset} color="error">
                        Reset
                    </Button>
                </div>
                <div style={{ display: 'flex', justifyContent: 'center', marginTop: '20px' }}>
                    <Button variant="outlined" color="success" onClick={handleSave}>
                        Save
                    </Button>
                </div>
            </CardContent>

            <Dialog open={open} onClose={handleClose} maxWidth="md" fullWidth={true}>
                <DialogTitle>Add Product</DialogTitle>
                <DialogContent>
                    <FormControl fullWidth margin="normal">
                        <InputLabel id="product-select-label">Product</InputLabel>
                        <Select
                            labelId="product-select-label"
                            value={selectedProduct}
                            onChange={(e) => setSelectedProduct(e.target.value)}
                        >
                            {products.map((product) => (
                                <MenuItem key={product.product_id} value={product.name}>
                                    <div>
                                        <Typography variant="body2">{product.name}</Typography>
                                        <Typography variant="body2" color="textSecondary">{`Price: ${product.unit_price} THB`}</Typography>
                                    </div>
                                </MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                    <TextField
                        label="Quantity"
                        type="number"
                        fullWidth
                        margin="normal"
                        value={quantity}
                        onChange={(e) => setQuantity(e.target.value)}
                    />
                </DialogContent>
                <DialogActions>
                    <Button onClick={handleClose} color="error">
                        Cancel
                    </Button>
                    <Button onClick={handleAddProduct} color="primary">
                        Add
                    </Button>
                </DialogActions>
            </Dialog>
        </Card>
    );
};

export default SalesForm;
