import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],

  // Prevent Vite from obscuring Rust errors
  clearScreen: false,

  server: {
    // Tauri expects a fixed port
    port: 1420,
    strictPort: true,
    // Allow connections from Tauri
    host: true,
  },

  // For transformers.js WASM support
  optimizeDeps: {
    exclude: ['@huggingface/transformers'],
  },

  build: {
    // Target modern browsers for WebGPU support
    target: 'esnext',
    // Tauri uses Chromium, safe to use modern features
    minify: 'esbuild',
  },

  // Handle WASM files
  assetsInclude: ['**/*.wasm'],

  // Worker configuration for transformers.js
  worker: {
    format: 'es',
  },
});
