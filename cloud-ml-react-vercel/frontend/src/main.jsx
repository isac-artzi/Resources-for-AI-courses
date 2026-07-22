/**
 * Application Entry Point
 * =======================
 *
 * This is the first file that runs when the React app starts.
 * It renders the root App component into the DOM.
 *
 * React 18 uses createRoot() for concurrent features.
 */

import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './App.css'

// Get the root DOM element where React will render
// This div is defined in index.html: <div id="root"></div>
const rootElement = document.getElementById('root')

// Create a React root using React 18's createRoot API
// This enables concurrent rendering features
const root = ReactDOM.createRoot(rootElement)

// Render the App component
// StrictMode helps identify potential problems during development
// It runs additional checks and warnings (development mode only)
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
