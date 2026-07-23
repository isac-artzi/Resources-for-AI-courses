# React + FastAPI Deep Learning App Tutorial

A complete guide to building, testing, and deploying a deep learning application with React frontend on Vercel and FastAPI backend on Render.

## Table of Contents
- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Local Development Setup](#local-development-setup)
- [Testing Locally](#testing-locally)
- [Deployment Guide](#deployment-guide)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)

## Project Overview

This tutorial demonstrates how to build a full-stack deep learning application that:
- Uses **FastAPI** for the backend with TensorFlow/PyTorch
- Uses **React** for the frontend with modern UI
- Implements image classification using a pre-trained deep learning model
- Allows local testing and cloud deployment
- Handles CORS for cross-origin requests
- Implements file upload and prediction

## Architecture

```
┌─────────────────┐         ┌─────────────────┐
│                 │         │                 │
│  React Frontend │ ◄─────► │ FastAPI Backend │
│   (Vercel)      │  HTTPS  │    (Render)     │
│                 │         │                 │
└─────────────────┘         └─────────────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │  Deep Learning  │
                            │     Model       │
                            │  (TensorFlow)   │
                            └─────────────────┘
```

## Prerequisites

### Required Software
- **Node.js** (v16 or higher) - [Download](https://nodejs.org/)
- **Python** (v3.8 or higher) - [Download](https://python.org/)
- **Git** - [Download](https://git-scm.com/)
- **npm** or **yarn** package manager
- **pip** Python package manager

### Accounts Needed
- **GitHub Account** - For code repository
- **Vercel Account** - For frontend deployment (free tier available)
- **Render Account** - For backend deployment (free tier available)

## Local Development Setup

### Step 1: Clone or Create Project

```bash
# Create project directory
mkdir react-fastapi-dl-app
cd react-fastapi-dl-app

# Initialize git repository
git init
```

### Step 2: Backend Setup (FastAPI)

```bash
# Create backend directory
mkdir backend
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Frontend Setup (React)

```bash
# Navigate back to root
cd ..

# Create React app
npx create-react-app frontend
cd frontend

# Install additional dependencies
npm install axios

# Navigate back to root
cd ..
```

## Testing Locally

### Step 1: Start the Backend

```bash
# Navigate to backend directory
cd backend

# Activate virtual environment if not already active
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Start FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at: `http://localhost:8000`
API documentation (Swagger UI): `http://localhost:8000/docs`

### Step 2: Start the Frontend

Open a new terminal:

```bash
# Navigate to frontend directory
cd frontend

# Start React development server
npm start
```

The frontend will be available at: `http://localhost:3000`

### Step 3: Test the Application

1. Open your browser to `http://localhost:3000`
2. Upload an image using the file input
3. Click "Classify Image"
4. View the prediction results

## Deployment Guide

### Part 1: Deploy Backend to Render

#### Step 1: Prepare Backend for Deployment

Ensure these files are in your `backend/` directory:
- `main.py` - FastAPI application
- `requirements.txt` - Python dependencies
- `render.yaml` (optional) - Render configuration

#### Step 2: Push to GitHub

```bash
# From project root
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

#### Step 3: Deploy on Render

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: `your-app-backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: `Free`
5. Add environment variables (if needed):
   - `PYTHON_VERSION`: `3.9.0`
6. Click **"Create Web Service"**

Wait for deployment to complete (5-10 minutes). Note your backend URL:
```
https://your-app-backend.onrender.com
```

### Part 2: Deploy Frontend to Vercel

#### Step 1: Update Frontend API URL

Edit `frontend/src/config.js`:

```javascript
// For production
export const API_URL = 'https://your-app-backend.onrender.com';
```

#### Step 2: Build and Deploy

```bash
# Navigate to frontend directory
cd frontend

# Install Vercel CLI (optional, can also use web interface)
npm install -g vercel

# Deploy using CLI
vercel

# Or deploy using Vercel Dashboard:
# 1. Go to https://vercel.com/dashboard
# 2. Click "Add New Project"
# 3. Import your GitHub repository
# 4. Configure:
#    - Framework Preset: Create React App
#    - Root Directory: frontend
#    - Build Command: npm run build
#    - Output Directory: build
# 5. Add Environment Variable:
#    - REACT_APP_API_URL: https://your-app-backend.onrender.com
# 6. Click "Deploy"
```

Your frontend will be deployed at:
```
https://your-app.vercel.app
```

#### Step 3: Update CORS Settings

Update `backend/main.py` to allow your Vercel domain:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://your-app.vercel.app"  # Add your Vercel URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Commit and push changes to trigger Render redeployment.

## Project Structure

```
react-fastapi-dl-app/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── model.py                # Deep learning model handler
│   ├── requirements.txt        # Python dependencies
│   ├── render.yaml            # Render configuration (optional)
│   └── venv/                  # Virtual environment (not committed)
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.js             # Main React component
│   │   ├── App.css            # Styles
│   │   ├── config.js          # API configuration
│   │   ├── components/
│   │   │   ├── ImageUpload.js # Image upload component
│   │   │   └── Results.js     # Results display component
│   │   └── index.js
│   ├── package.json
│   └── .env                   # Environment variables
├── .gitignore
└── README.md                  # This file
```

## API Documentation

### Endpoints

#### `GET /`
Health check endpoint.

**Response:**
```json
{
  "message": "Welcome to Deep Learning API",
  "status": "active"
}
```

#### `POST /predict`
Upload an image and get classification predictions.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response:**
```json
{
  "predictions": [
    {
      "class": "golden_retriever",
      "confidence": 0.95
    },
    {
      "class": "labrador",
      "confidence": 0.03
    }
  ]
}
```

#### `GET /health`
Server health status.

**Response:**
```json
{
  "status": "healthy"
}
```

## Environment Variables

### Backend (.env)
```
PORT=8000
ENVIRONMENT=development
```

### Frontend (.env)
```
REACT_APP_API_URL=http://localhost:8000
```

## Troubleshooting

### Common Issues

**Issue 1: CORS Error**
- Ensure backend CORS middleware includes your frontend URL
- Check that API_URL in frontend is correct

**Issue 2: Model Loading Error**
- Ensure TensorFlow is installed correctly
- Check Python version compatibility
- Verify model file exists

**Issue 3: Deployment Fails on Render**
- Check build logs in Render dashboard
- Verify requirements.txt has all dependencies
- Ensure start command is correct

**Issue 4: Vercel Build Fails**
- Check that build command is `npm run build`
- Verify output directory is `build`
- Check for any missing dependencies

## Next Steps

1. **Add Authentication**: Implement user authentication
2. **Database Integration**: Add PostgreSQL for storing predictions
3. **Model Versioning**: Implement model version management
4. **Monitoring**: Add logging and monitoring with Sentry
5. **Custom Models**: Train and deploy your own models
6. **Batch Processing**: Support multiple image uploads

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Vercel Documentation](https://vercel.com/docs)
- [Render Documentation](https://render.com/docs)
- [TensorFlow Documentation](https://www.tensorflow.org/)

## License

MIT License - Feel free to use this tutorial for learning and commercial projects.

## Support

For issues and questions:
- Check the troubleshooting section
- Review deployment logs
- Open an issue on GitHub
