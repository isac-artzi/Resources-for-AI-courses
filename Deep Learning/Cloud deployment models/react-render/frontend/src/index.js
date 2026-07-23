/**
 * React Application Entry Point
 *
 * This is the first JavaScript file that runs when the app starts.
 * It connects React to the HTML page and renders the App component.
 *
 * Key Concepts:
 * - ReactDOM: Library for rendering React components to the DOM
 * - StrictMode: Development tool that checks for problems
 * - Root: The HTML element where React will render
 */

// Import React and ReactDOM libraries
import React from 'react';
import ReactDOM from 'react-dom/client';

// Import global styles
import './index.css';

// Import the main App component
import App from './App';

// Import web vitals for performance monitoring (optional)
import reportWebVitals from './reportWebVitals';

/**
 * Create a React root and render the app
 *
 * Steps:
 * 1. Find the 'root' div in index.html
 * 2. Create a React root from that element
 * 3. Render the App component inside StrictMode
 */

// Get the root DOM element from index.html
const rootElement = document.getElementById('root');

// Create a React root
// This is the new way in React 18+ (previously used ReactDOM.render)
const root = ReactDOM.createRoot(rootElement);

// Render the application
root.render(
  // StrictMode is a development tool that:
  // - Detects potential problems in the app
  // - Warns about deprecated features
  // - Does NOT affect production builds
  <React.StrictMode>
    {/* The main App component - everything starts here */}
    <App />
  </React.StrictMode>
);

/**
 * Web Vitals (Performance Monitoring)
 *
 * This measures and reports performance metrics like:
 * - LCP: Largest Contentful Paint (loading performance)
 * - FID: First Input Delay (interactivity)
 * - CLS: Cumulative Layout Shift (visual stability)
 *
 * You can send these metrics to an analytics service
 * For now, we'll just log them to console
 */
reportWebVitals(console.log);

// Alternative: Send to analytics service
// reportWebVitals(sendToAnalytics);
