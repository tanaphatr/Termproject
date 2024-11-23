import React, { createContext, useState } from 'react';

export const SalesContext = createContext();

export const SalesProvider = ({ children }) => {
    const [salesData, setSalesData] = useState([]);

    return (
        <SalesContext.Provider value={{ salesData, setSalesData }}>
            {children}
        </SalesContext.Provider>
    );
};
