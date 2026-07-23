/**
 * index.js - React Application Entry Point
 *
 * This file is the first JavaScript file that runs when your app loads.
 * It's responsible for:
 * 1. Importing React and ReactDOM libraries
 * 2. Importing our main App component
 * 3. Rendering the App component into the HTML DOM
 */

import React from 'react';
import ReactDOM from 'react-dom/client';
import './App.css';
import App from './App';

// Get the root DOM element from public/index.html
// This is where our entire React app will be inserted
const root = ReactDOM.createRoot(document.getElementById('root'));

// Render our App component inside React.StrictMode
// StrictMode helps identify potential problems in the app during development
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
