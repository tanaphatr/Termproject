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
                product_id: product.product_id, // Using product_id consistently
                name: product.name,
                quantity: qty,
                unit_price: product.unit_price,
                total: product.unit_price * qty,
            };
            console.log("Adding product:", newProduct); // Log the product being added
            setAddedProducts(prevProducts => [...prevProducts, newProduct]);
            handleClose();
        } else {
            alert("Please select a product and enter a valid quantity.");
        }
    };

    const handleRemoveProduct = (product_id) => {
        console.log("Removing product with id:", product_id); // Log the product ID being removed
        setAddedProducts(prevProducts => prevProducts.filter(prod => prod.product_id !== product_id));
    };

    const handleSave = () => {
        console.log("Saving products:", addedProducts);
        // Add your code to save the data here
    };

    const handleReset = () => {
        setAddedProducts([]); // Clear added products
        setSelectedProduct(''); // Reset selected product
        setQuantity(''); // Reset quantity
    };

    return (
        <Card style={{ padding: '20px' }}>
            <CardContent>
                <Typography variant="h5" align="center">Sales Form</Typography>
                <TableContainer component={Paper} style={{ marginTop: '20px' }}>
                    <Table>
                        <TableHead>
                            <TableRow>
                                <TableCell>Nameproduct</TableCell>
                                <TableCell>Quantity</TableCell>
                                <TableCell>Unitprice</TableCell>
                                <TableCell>Total</TableCell>
                                <TableCell>Action</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {addedProducts.map((product) => (
                                <TableRow key={product.product_id}>
                                    <TableCell>{product.name}</TableCell>
                                    <TableCell>{product.quantity}</TableCell>
                                    <TableCell>{product.unit_price}</TableCell>
                                    <TableCell>{product.total}</TableCell>
                                    <TableCell>
                                        <Button
                                            variant="outlined"
                                            color="error"
                                            onClick={() => handleRemoveProduct(product.product_id)} // Send the product_id to remove
                                        >
                                            Delete
                                        </Button>
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </TableContainer>
                <div style={{ display: 'flex', justifyContent: 'center', marginTop: '20px' }}>
                    <Button variant="outlined" color="primary" onClick={handleClickOpen}>
                        Add Product
                    </Button>
                </div>
                <div style={{ display: 'flex', justifyContent: 'center', marginTop: '20px' }}>
                    <Button variant="outlined" onClick={handleSave} color="success">
                        Save
                    </Button>
                    <Button variant="outlined" onClick={handleReset} color="error" style={{ marginLeft: '10px' }}>
                        Reset
                    </Button>
                </div>
            </CardContent>

            <Dialog open={open} onClose={handleClose}>
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
                                    {product.name}
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
