# React + FastAPI Deep Learning Tutorial
## Deploy on Vercel

This tutorial teaches you how to build a full-stack deep learning application with:
- **Frontend**: React (with Vite)
- **Backend**: FastAPI (Python)
- **Deep Learning**: TensorFlow/Keras for image classification
- **Deployment**: Vercel (frontend) + Railway/Render (backend)

## 📚 Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Project Structure](#project-structure)
5. [Setup Instructions](#setup-instructions)
6. [Local Development](#local-development)
7. [Deployment Guide](#deployment-guide)
8. [API Documentation](#api-documentation)
9. [Troubleshooting](#troubleshooting)

## 🎯 Project Overview

This application demonstrates a complete deep learning workflow:
- Users upload images through a React interface
- Images are sent to a FastAPI backend
- A pre-trained MobileNetV2 model classifies the images
- Results are displayed in real-time with confidence scores

**What You'll Learn:**
- Building modern React applications with hooks
- Creating RESTful APIs with FastAPI
- Integrating deep learning models into web apps
- Handling file uploads and CORS
- Deploying full-stack applications
- Environment configuration for production

## 🏗️ Architecture

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│                 │         │                 │         │                 │
│  React Frontend │────────▶│  FastAPI Backend│────────▶│  ML Model       │
│  (Vercel)       │  HTTP   │  (Railway)      │         │  (MobileNetV2)  │
│                 │◀────────│                 │◀────────│                 │
└─────────────────┘  JSON   └─────────────────┘         └─────────────────┘
```

**Frontend (React):**
- File upload interface
- Image preview
- Results display
- Loading states and error handling

**Backend (FastAPI):**
- `/predict` endpoint for image classification
- CORS middleware for cross-origin requests
- Image processing and validation
- Model inference

**Deep Learning:**
- Pre-trained MobileNetV2 (ImageNet)
- Image preprocessing pipeline
- Top-5 predictions with confidence scores

## ✅ Prerequisites

Before starting, ensure you have:

- **Node.js** (v18 or higher) - [Download](https://nodejs.org/)
- **Python** (v3.8 or higher) - [Download](https://python.org/)
- **Git** - [Download](https://git-scm.com/)
- **Vercel Account** - [Sign up](https://vercel.com/)
- **Railway Account** (for backend) - [Sign up](https://railway.app/)
- **Code Editor** (VS Code recommended)

**Check your installations:**
```bash
node --version    # Should be v18+
python --version  # Should be 3.8+
git --version
```

## 📁 Project Structure

```
cloud-ml-react-vercel/
├── frontend/                  # React application
│   ├── public/               # Static files
│   ├── src/
│   │   ├── components/       # React components
│   │   │   ├── ImageUpload.jsx
│   │   │   ├── Results.jsx
│   │   │   └── LoadingSpinner.jsx
│   │   ├── services/         # API integration
│   │   │   └── api.js
│   │   ├── App.jsx           # Main application
│   │   ├── App.css           # Styles
│   │   └── main.jsx          # Entry point
│   ├── package.json
│   ├── vite.config.js
│   └── .env.local            # Environment variables
│
├── backend/                   # FastAPI application
│   ├── app/
│   │   ├── main.py           # FastAPI app & routes
│   │   ├── model.py          # ML model wrapper
│   │   └── utils.py          # Helper functions
│   ├── requirements.txt      # Python dependencies
│   ├── runtime.txt           # Python version
│   └── .env                  # Environment variables
│
├── README.md                 # This file
└── DEPLOYMENT.md             # Detailed deployment guide
```

## 🚀 Setup Instructions

### Step 1: Clone/Create Project Directory

```bash
# If you haven't already, navigate to your project directory
cd cloud-ml-react-vercel
```

### Step 2: Setup Backend (FastAPI)

```bash
# Create backend directory
mkdir -p backend/app
cd backend

# Create Python virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies (we'll create requirements.txt first)
pip install fastapi uvicorn python-multipart pillow tensorflow numpy python-dotenv

# Save dependencies
pip freeze > requirements.txt

# Go back to root
cd ..
```

### Step 3: Setup Frontend (React)

```bash
# Create React app with Vite
npm create vite@latest frontend -- --template react

# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Install additional packages
npm install axios

# Go back to root
cd ..
```

## 💻 Local Development

### Running the Backend

```bash
# Navigate to backend directory
cd backend

# Activate virtual environment (if not already active)
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Run FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

### Running the Frontend

```bash
# In a NEW terminal, navigate to frontend directory
cd frontend

# Start development server
npm run dev
```

The frontend will be available at: `http://localhost:5173`

### Testing the Application

1. Open `http://localhost:5173` in your browser
2. Click "Choose Image" or drag & drop an image
3. Click "Classify Image"
4. View the predictions with confidence scores

**Test Images:** Use any `.jpg`, `.jpeg`, or `.png` image. Try:
- Animals (dogs, cats, birds)
- Objects (cars, phones, furniture)
- Food items

## 🌐 Deployment Guide

### Deploying the Backend (Railway)

**Why Railway?** Vercel serverless functions have limitations for heavy ML models. Railway provides persistent servers ideal for deep learning applications.

1. **Prepare Backend for Deployment:**
   ```bash
   cd backend
   # Ensure requirements.txt is up to date
   pip freeze > requirements.txt
   ```

2. **Create Railway Project:**
   - Go to [Railway](https://railway.app/)
   - Click "New Project" → "Deploy from GitHub repo"
   - Connect your GitHub repository
   - Select the repository
   - Railway will auto-detect Python

3. **Configure Railway:**
   - Add root directory: `backend`
   - Add start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Set environment variables (if needed)

4. **Deploy:**
   - Railway will automatically build and deploy
   - Copy the provided URL (e.g., `https://your-app.railway.app`)

### Deploying the Frontend (Vercel)

1. **Update API URL:**
   ```bash
   cd frontend
   # Create .env.production file
   echo "VITE_API_URL=https://your-app.railway.app" > .env.production
   ```

2. **Build Frontend:**
   ```bash
   npm run build
   ```

3. **Deploy to Vercel:**

   **Option A: Vercel CLI**
   ```bash
   # Install Vercel CLI
   npm i -g vercel

   # Login to Vercel
   vercel login

   # Deploy
   vercel --prod
   ```

   **Option B: Vercel Dashboard**
   - Go to [Vercel](https://vercel.com/)
   - Click "New Project"
   - Import your Git repository
   - Set root directory to `frontend`
   - Add environment variable: `VITE_API_URL=https://your-app.railway.app`
   - Click "Deploy"

4. **Configure Build Settings:**
   - Build Command: `npm run build`
   - Output Directory: `dist`
   - Install Command: `npm install`

### Alternative: Deploy Backend on Render

1. Create account on [Render](https://render.com/)
2. Create new "Web Service"
3. Connect GitHub repository
4. Configure:
   - Root Directory: `backend`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Deploy and copy URL

## 📖 API Documentation

### Endpoints

#### `GET /`
- **Description:** Health check endpoint
- **Response:** `{"message": "FastAPI Deep Learning API", "status": "running"}`

#### `POST /predict`
- **Description:** Classify uploaded image
- **Request:**
  - Content-Type: `multipart/form-data`
  - Body: `file` (image file)
- **Response:**
  ```json
  {
    "success": true,
    "predictions": [
      {"class": "golden_retriever", "confidence": 0.89},
      {"class": "Labrador_retriever", "confidence": 0.06}
    ],
    "processing_time": 0.234
  }
  ```

#### `GET /health`
- **Description:** Detailed health check
- **Response:**
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "timestamp": "2024-01-15T10:30:00"
  }
  ```

## 🐛 Troubleshooting

### Backend Issues

**Problem:** `ModuleNotFoundError: No module named 'tensorflow'`
```bash
# Solution: Ensure virtual environment is activated and dependencies installed
source venv/bin/activate
pip install -r requirements.txt
```

**Problem:** CORS errors in browser
```bash
# Solution: Check CORS configuration in backend/app/main.py
# Ensure frontend URL is in allowed origins
```

**Problem:** Model loading is slow
```bash
# This is normal on first run - TensorFlow downloads MobileNetV2 (~14MB)
# Subsequent runs will use cached model
```

### Frontend Issues

**Problem:** API connection refused
```bash
# Solution: Ensure backend is running on port 8000
# Check .env.local has correct VITE_API_URL
```

**Problem:** Build fails on Vercel
```bash
# Solution: Ensure package.json is in frontend directory
# Check Node version compatibility (use v18+)
```

### Deployment Issues

**Problem:** Railway build fails
```bash
# Check Python version in runtime.txt (should be 3.8+)
# Verify all dependencies in requirements.txt
# Check logs in Railway dashboard
```

**Problem:** Out of memory on Railway
```bash
# TensorFlow can be memory-intensive
# Upgrade Railway plan or use lighter model
# Consider using TensorFlow Lite
```

## 🎓 Learning Resources

- **React:** [Official React Docs](https://react.dev/)
- **FastAPI:** [FastAPI Documentation](https://fastapi.tiangolo.com/)
- **TensorFlow:** [TensorFlow Guides](https://www.tensorflow.org/guide)
- **Vercel:** [Vercel Documentation](https://vercel.com/docs)
- **Railway:** [Railway Documentation](https://docs.railway.app/)

## 📝 Next Steps

After completing this tutorial, try:

1. **Custom Models:** Train your own model and integrate it
2. **Multiple Models:** Add model selection in the UI
3. **Advanced Features:**
   - Image preprocessing options
   - Batch processing
   - Result history
4. **Database:** Add MongoDB/PostgreSQL to store predictions
5. **Authentication:** Add user authentication with Auth0 or Clerk
6. **Real-time:** Implement WebSocket for live predictions

## 📄 License

This tutorial is for educational purposes.

## 🤝 Contributing

Students are encouraged to:
- Report issues
- Suggest improvements
- Share their deployed applications

---

**Happy Coding! 🚀**

For questions, please refer to the course materials or contact your instructor.
