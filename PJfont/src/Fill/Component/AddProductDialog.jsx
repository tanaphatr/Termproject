import React from 'react';
import {
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
    Button,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    TextField,
    Typography,
} from '@mui/material';

const AddProductDialog = ({ open, handleClose, handleAddProduct, products, selectedProduct, setSelectedProduct, quantity, setQuantity }) => {
    return (
        <Dialog open={open} onClose={handleClose} maxWidth="md" fullWidth={true}>
            <DialogTitle>Add Product</DialogTitle>
            <DialogContent>
                <FormControl fullWidth margin="normal">
                    <InputLabel id="product-select-label">Product</InputLabel>
                    <Select
                        labelId="product-select-label"
                        value={selectedProduct}
                        onChange={(e) => setSelectedProduct(e.target.value)}
                        MenuProps={{
                            PaperProps: {
                                style: {
                                    maxHeight: 200,
                                    width: 300,
                                },
                            },
                        }}
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
    );
};

export default AddProductDialog;
