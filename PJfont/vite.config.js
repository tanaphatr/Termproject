import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path'; // เพิ่มการนำเข้า path

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'), // ตั้งค่า alias สำหรับ src
    },
  },
});
