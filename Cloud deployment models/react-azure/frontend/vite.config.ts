/**
 * Vite Configuration File
 *
 * Vite is a build tool that provides fast development server
 * and optimized production builds.
 */

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],

  // Development server configuration
  server: {
    port: 5173,
    host: true, // Listen on all addresses
    strictPort: false, // Try next port if 5173 is taken
  },

  // Build configuration
  build: {
    outDir: 'dist',
    sourcemap: true, // Generate source maps for debugging
    // Optimize chunks
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom'],
        },
      },
    },
  },

  // Environment variable prefix
  // Only variables starting with VITE_ are exposed to the client
  envPrefix: 'VITE_',
})
