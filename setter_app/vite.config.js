import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import wasm from 'vite-plugin-wasm';
import legacy from '@vitejs/plugin-legacy';

export default defineConfig({
  base: "/setter",
  plugins: [
    react(),
    wasm(),
    legacy({
      targets: ['defaults', 'not IE 11'],
    }),
  ],
  optimizeDeps: {
    exclude: ['onnxruntime-web'],  // Disable pre-bundling for ONNX
  },
  server: {
    fs: {
      strict: false, // Allow serving files from root
    },
    proxy: {
      '/api': {
        target: 'https://baritodespacito.pythonanywhere.com',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
        secure: false
      }
    },
  },
});