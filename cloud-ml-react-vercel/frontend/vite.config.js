/**
 * Vite Configuration
 * ==================
 *
 * Vite is a modern build tool that provides:
 * - Lightning-fast dev server with Hot Module Replacement (HMR)
 * - Optimized production builds
 * - Built-in TypeScript/JSX support
 *
 * This configuration sets up React and defines build settings.
 */

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  // Plugins
  plugins: [react()],

  // Development server configuration
  server: {
    port: 5173,           // Default Vite port
    host: true,           // Listen on all addresses (0.0.0.0)
    open: true,           // Auto-open browser on server start

    // Proxy API requests to backend during development
    // This avoids CORS issues in development
    proxy: {
      // Uncomment if you want to proxy API calls through Vite
      // '/api': {
      //   target: 'http://localhost:8000',
      //   changeOrigin: true,
      //   rewrite: (path) => path.replace(/^\/api/, '')
      // }
    }
  },

  // Build configuration for production
  build: {
    outDir: 'dist',       // Output directory for build files
    sourcemap: true,      // Generate source maps for debugging

    // Optimize chunk size warnings
    chunkSizeWarningLimit: 1000,

    // Rollup options for advanced build customization
    rollupOptions: {
      output: {
        // Manual chunk splitting for better caching
        manualChunks: {
          'react-vendor': ['react', 'react-dom'],
          'utils': ['axios']
        }
      }
    }
  },

  // Environment variable prefix
  // Only variables starting with VITE_ are exposed to the app
  envPrefix: 'VITE_'
})
