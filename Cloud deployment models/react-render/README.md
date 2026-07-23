# React + FastAPI Deep Learning Tutorial

## 📚 Tutorial Overview

This tutorial teaches you how to build a full-stack deep learning application with:
- **Frontend**: React.js for the user interface
- **Backend**: FastAPI for serving deep learning models
- **Deep Learning**: Image classification using PyTorch
- **Deployment**: Render.com for cloud deployment

### What You'll Build
An image classification web app where users can upload images and get AI-powered predictions.

---

## 🏗️ Project Structure

```
react-render/
├── backend/                    # FastAPI backend
│   ├── app.py                 # Main FastAPI application
│   ├── model.py               # Deep learning model
│   ├── requirements.txt       # Python dependencies
│   └── render.yaml           # Render deployment config
├── frontend/                  # React frontend
│   ├── public/               # Static files
│   ├── src/                  # React source code
│   │   ├── App.js           # Main React component
│   │   ├── components/      # React components
│   │   └── services/        # API services
│   └── package.json         # Node dependencies
└── README.md                # This file
```

---

## 🎓 Learning Objectives

By completing this tutorial, you will learn:

1. **Backend Development**
   - Create RESTful APIs with FastAPI
   - Integrate deep learning models (PyTorch)
   - Handle file uploads and processing
   - Implement CORS for cross-origin requests

2. **Frontend Development**
   - Build interactive React components
   - Handle file uploads in React
   - Make API calls with fetch/axios
   - Display dynamic results

3. **Deep Learning Integration**
   - Load pre-trained models
   - Preprocess images for neural networks
   - Make predictions and return results

4. **Cloud Deployment**
   - Deploy FastAPI backend on Render
   - Deploy React frontend on Render
   - Configure environment variables
   - Connect frontend and backend services

---

## 🚀 Quick Start Guide

### Prerequisites
- Node.js (v14 or higher)
- Python (3.8 or higher)
- Git
- Code editor (VS Code recommended)

### Step 1: Clone and Setup

```bash
# Navigate to project directory
cd react-render

# Setup backend
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Setup frontend (in a new terminal)
cd frontend
npm install
```

### Step 2: Run Locally

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
uvicorn app:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```

Visit `http://localhost:3000` in your browser!

---

## 📖 Detailed Tutorial

### Part 1: Understanding the Backend (FastAPI)

#### What is FastAPI?
FastAPI is a modern, fast Python web framework for building APIs. It's perfect for serving machine learning models because:
- **Fast**: High performance, comparable to Node.js
- **Easy**: Simple syntax with automatic documentation
- **Type-safe**: Uses Python type hints
- **Async**: Supports asynchronous operations

#### The Backend Structure

1. **app.py**: Main application file
   - Defines API endpoints (routes)
   - Handles image uploads
   - Returns predictions

2. **model.py**: Deep learning model wrapper
   - Loads the pre-trained model
   - Preprocesses images
   - Makes predictions

#### Key Concepts:

**API Endpoint**: A URL that clients can call to interact with the server
```python
@app.post("/predict")  # This creates a POST endpoint at /predict
```

**CORS**: Cross-Origin Resource Sharing - allows frontend to call backend
```python
app.add_middleware(CORSMiddleware)  # Enables cross-origin requests
```

**File Upload**: Handling image uploads from users
```python
file: UploadFile = File(...)  # Receives uploaded files
```

---

### Part 2: Understanding the Frontend (React)

#### What is React?
React is a JavaScript library for building user interfaces. Key concepts:

1. **Components**: Reusable UI pieces (like LEGO blocks)
2. **State**: Data that changes over time
3. **Props**: Data passed between components
4. **Hooks**: Functions to add features (useState, useEffect)

#### The Frontend Structure

1. **App.js**: Main component
   - Manages application state
   - Coordinates child components

2. **ImageUpload.js**: Upload component
   - Handles file selection
   - Sends images to backend

3. **ResultDisplay.js**: Results component
   - Shows predictions
   - Displays confidence scores

#### Key React Concepts:

**State Management**:
```javascript
const [image, setImage] = useState(null);  // Stores uploaded image
const [result, setResult] = useState(null); // Stores prediction
```

**Event Handling**:
```javascript
const handleImageChange = (e) => {
  // Responds to user actions
}
```

**API Calls**:
```javascript
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData
});
```

---

### Part 3: Deep Learning Integration

#### The Model
We use a pre-trained ResNet model from PyTorch:
- **ResNet**: Residual Network, excellent for image classification
- **Pre-trained**: Already trained on ImageNet (1000 categories)
- **Transfer Learning**: Use existing knowledge for our task

#### Image Processing Pipeline:

1. **Upload**: User selects image
2. **Send**: Frontend sends to backend
3. **Preprocess**: Resize, normalize, convert to tensor
4. **Predict**: Model processes image
5. **Return**: Send top predictions to frontend
6. **Display**: Show results to user

---

### Part 4: Testing Locally

#### Backend Testing

1. **Test API Health**:
```bash
curl http://localhost:8000/
```

2. **Test with Swagger Docs**:
   - Visit `http://localhost:8000/docs`
   - Interactive API documentation
   - Try uploading images directly

3. **Test Prediction Endpoint**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@test_image.jpg"
```

#### Frontend Testing

1. **Check React App**: Visit `http://localhost:3000`
2. **Upload Image**: Select an image file
3. **View Results**: Check predictions appear
4. **Check Console**: Open browser DevTools (F12)

#### Common Issues & Solutions:

**CORS Error**:
- Backend not running
- CORS not configured properly
- Check browser console

**Connection Refused**:
- Backend might not be running on port 8000
- Check terminal for errors

**Prediction Failed**:
- Image format not supported
- Model not loaded properly
- Check backend logs

---

## 🌐 Deployment on Render

### Why Render?
- Free tier available
- Easy deployment
- Automatic HTTPS
- Good for student projects

### Prerequisites for Deployment

1. **GitHub Account**: Store code
2. **Render Account**: Sign up at render.com
3. **Code Pushed to GitHub**:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin main
```

---

### Deploying the Backend

#### Step 1: Create Web Service on Render

1. Go to [render.com](https://render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: `your-app-backend`
   - **Region**: Choose closest to you
   - **Branch**: `main`
   - **Root Directory**: `backend`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`

#### Step 2: Environment Variables

Add these in Render dashboard:
- `PYTHON_VERSION`: `3.9.0`

#### Step 3: Deploy

- Click "Create Web Service"
- Wait for build (3-5 minutes)
- Note your backend URL: `https://your-app-backend.onrender.com`

---

### Deploying the Frontend

#### Step 1: Update API URL

In `frontend/src/services/api.js`, change:
```javascript
const API_URL = 'https://your-app-backend.onrender.com';
```

#### Step 2: Create Static Site on Render

1. Click "New +" → "Static Site"
2. Connect repository
3. Configure:
   - **Name**: `your-app-frontend`
   - **Branch**: `main`
   - **Root Directory**: `frontend`
   - **Build Command**: `npm install && npm run build`
   - **Publish Directory**: `build`

#### Step 3: Environment Variables (if needed)

```
REACT_APP_API_URL=https://your-app-backend.onrender.com
```

#### Step 4: Deploy

- Click "Create Static Site"
- Wait for build
- Visit your frontend URL: `https://your-app-frontend.onrender.com`

---

### Post-Deployment Checklist

✅ Backend is accessible at its Render URL
✅ Frontend loads without errors
✅ Image upload works
✅ Predictions are displayed
✅ No CORS errors in console
✅ HTTPS is working

---

## 🔧 Troubleshooting

### Backend Issues

**"Module not found" Error**:
```bash
# Make sure all dependencies are in requirements.txt
pip freeze > requirements.txt
```

**Model Download Timeout**:
```python
# Model downloads on first run - this is normal
# Render free tier might timeout
# Solution: Use smaller model or upgrade plan
```

### Frontend Issues

**API Call Fails**:
- Check API URL is correct
- Verify backend is running
- Check browser console for errors

**Build Fails**:
- Clear npm cache: `npm cache clean --force`
- Delete node_modules: `rm -rf node_modules`
- Reinstall: `npm install`

### Deployment Issues

**Render Build Fails**:
- Check build logs in Render dashboard
- Verify directory structure
- Ensure all files are committed to Git

**App Crashes on Render**:
- Check application logs
- Verify environment variables
- Check memory usage (free tier limits)

---

## 🎯 Learning Exercises

### Beginner
1. Change the welcome message in the frontend
2. Modify CSS to change colors
3. Add a loading spinner during prediction

### Intermediate
1. Add support for multiple image formats
2. Display top 5 predictions instead of top 3
3. Add error handling for invalid images

### Advanced
1. Implement image history (store previous predictions)
2. Add user authentication
3. Train a custom model for specific categories
4. Add batch image upload

---

## 📚 Additional Resources

### FastAPI
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)

### React
- [React Documentation](https://react.dev/)
- [React Tutorial](https://react.dev/learn)

### PyTorch
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Transfer Learning Guide](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

### Deployment
- [Render Documentation](https://render.com/docs)
- [Deploying Python Apps](https://render.com/docs/deploy-fastapi)

---

## 🤝 Contributing

This is a tutorial project. You can:
- Fork the repository
- Try different models
- Improve the UI
- Add new features
- Share improvements

---

## 📝 License

This tutorial is for educational purposes.

---

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section
2. Review error messages carefully
3. Search for errors online
4. Ask your instructor or classmates

---

## 🎉 Congratulations!

You've learned how to:
✅ Build a FastAPI backend
✅ Create a React frontend
✅ Integrate deep learning models
✅ Deploy to the cloud

Keep building and learning! 🚀
