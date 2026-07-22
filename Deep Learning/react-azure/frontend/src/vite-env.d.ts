/// <reference types="vite/client" />

/**
 * Type definitions for environment variables
 * This provides TypeScript autocomplete and type checking for import.meta.env
 */

interface ImportMetaEnv {
  readonly VITE_API_URL: string
  // Add more environment variables here as needed
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
