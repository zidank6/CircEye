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
    // Headers for WASM and cross-origin isolation
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'credentialless',
    },
  },

  // For transformers.js WASM support - don't pre-bundle
  optimizeDeps: {
    exclude: ['@huggingface/transformers'],
  },

  build: {
    // Target modern browsers
    target: 'esnext',
    minify: 'esbuild',
  },

  // Handle WASM files
  assetsInclude: ['**/*.wasm', '**/*.onnx'],

  // Worker configuration for transformers.js
  worker: {
    format: 'es',
  },
});
