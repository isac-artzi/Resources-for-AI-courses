# 🧪 Testing Guide

Complete guide for testing your React + FastAPI application locally before deployment.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Backend Testing](#backend-testing)
3. [Frontend Testing](#frontend-testing)
4. [Integration Testing](#integration-testing)
5. [Troubleshooting](#troubleshooting)
6. [Common Issues](#common-issues)

---

## Prerequisites

### Required Software

✅ **Python 3.8 or higher**
```bash
python --version
# or
python3 --version
```

✅ **Node.js 14 or higher**
```bash
node --version
npm --version
```

✅ **Git** (for version control)
```bash
git --version
```

### Test Image

You'll need at least one test image. Download sample images from:
- [Unsplash](https://unsplash.com/) (free high-quality images)
- [Pexels](https://www.pexels.com/) (free stock photos)
- Use any image from your computer

Recommended test images:
- Cat or dog (pets are well-recognized)
- Common objects (car, laptop, coffee cup)
- Animals (bird, elephant, tiger)

---

## Backend Testing

### Step 1: Setup Backend Environment

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Verify activation (should show (venv) in prompt)
```

### Step 2: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# This will install:
# - fastapi
# - uvicorn
# - python-multipart
# - Pillow
# - torch
# - torchvision

# Note: This may take 3-5 minutes (PyTorch is large)
```

### Step 3: Verify Installation

```bash
# Check installed packages
pip list

# Should see all packages from requirements.txt
```

### Step 4: Test Model Loading

```bash
# Run the model.py file directly
python model.py

# This will:
# - Load the ResNet-18 model
# - Download pre-trained weights (first time only)
# - Test basic functionality

# Expected output:
# "Loading pre-trained ResNet-18 model..."
# "Model loaded successfully!"
```

### Step 5: Start Backend Server

```bash
# Start the FastAPI server
uvicorn app:app --reload --port 8000

# Expected output:
# INFO:     Uvicorn running on http://127.0.0.1:8000
# INFO:     Application startup complete.

# The server is now running!
```

### Step 6: Test Backend Endpoints

**Open a NEW terminal** (keep the server running in the first terminal)

#### Test 1: Root Endpoint

```bash
# Test the root endpoint
curl http://localhost:8000/

# Expected response:
# {
#   "message": "Welcome to the Image Classification API!",
#   "status": "running",
#   ...
# }
```

#### Test 2: Health Check

```bash
# Test health endpoint
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "model_loaded": true,
#   ...
# }
```

#### Test 3: Model Info

```bash
# Get model information
curl http://localhost:8000/model-info

# Expected response:
# {
#   "model_name": "ResNet-18",
#   "framework": "PyTorch",
#   ...
# }
```

#### Test 4: Interactive API Documentation

1. Open browser
2. Go to: `http://localhost:8000/docs`
3. You should see Swagger UI with all endpoints
4. Try uploading an image using the interface:
   - Click on `/predict` endpoint
   - Click "Try it out"
   - Click "Choose File" and select an image
   - Click "Execute"
   - Check the response

#### Test 5: Image Upload (Command Line)

```bash
# Upload an image using curl
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/your/image.jpg"

# Replace path/to/your/image.jpg with actual image path

# Expected response:
# {
#   "success": true,
#   "filename": "image.jpg",
#   "predictions": [
#     {
#       "label": "tabby cat",
#       "confidence": 0.92,
#       "class_id": 281
#     },
#     ...
#   ]
# }
```

### Backend Test Checklist

✅ Virtual environment activated
✅ All dependencies installed
✅ Model loads without errors
✅ Server starts successfully
✅ Root endpoint responds
✅ Health check passes
✅ Can upload and classify images
✅ Predictions look reasonable

---

## Frontend Testing

### Step 1: Setup Frontend Environment

**Open a NEW terminal** (keep backend running)

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# This installs all packages from package.json
# May take 2-3 minutes
```

### Step 2: Verify Installation

```bash
# Check installed packages
npm list --depth=0

# Should see React and other dependencies
```

### Step 3: Start Development Server

```bash
# Start React development server
npm start

# Expected output:
# Compiled successfully!
#
# You can now view the app in the browser.
# Local: http://localhost:3000

# Browser should open automatically
# If not, manually open: http://localhost:3000
```

### Step 4: Visual Testing

#### Test 1: Initial Load

✅ Page loads without errors
✅ Title displays: "🤖 AI Image Classifier"
✅ Backend status shows "Backend Online" (green badge)
✅ Upload area is visible
✅ No console errors (press F12 to check)

#### Test 2: Image Selection

1. Click on the upload area
2. Select an image file
3. ✅ Preview appears
4. ✅ Filename displays
5. ✅ File size shows
6. ✅ "Classify Image" button appears

#### Test 3: Image Upload

1. Click "Classify Image" button
2. ✅ Button shows "Classifying..." with spinner
3. ✅ Button is disabled during upload
4. ✅ Results appear after processing (2-5 seconds)
5. ✅ Top predictions display with confidence scores
6. ✅ Confidence bars show correctly

#### Test 4: Results Display

✅ Success icon appears (green checkmark)
✅ Predictions are ranked (1, 2, 3, ...)
✅ Labels are capitalized
✅ Confidence percentages are correct
✅ Confidence bars match percentages
✅ Colors vary based on confidence level

#### Test 5: Try Another Image

1. Click "Classify Another Image" button
2. ✅ Form resets
3. ✅ Previous results clear
4. ✅ Can upload new image

#### Test 6: Error Handling

**Stop the backend server** (Ctrl+C in backend terminal)

1. Try to upload an image
2. ✅ Error message appears
3. ✅ Error is user-friendly
4. ✅ Suggests checking backend

**Restart the backend server**

### Step 5: Browser Console Testing

Press **F12** to open Developer Tools

#### Check Console

✅ No errors (red messages)
✅ API URL is logged correctly
✅ Upload process logs appear
✅ Successful predictions logged

#### Check Network Tab

1. Go to Network tab
2. Upload an image
3. ✅ See POST request to `/predict`
4. ✅ Status: 200 OK
5. ✅ Response contains predictions
6. ✅ Request shows FormData with file

### Frontend Test Checklist

✅ npm install completes successfully
✅ Development server starts
✅ Page loads without errors
✅ Backend connection works
✅ Image selection works
✅ Image preview displays
✅ Upload and classification work
✅ Results display correctly
✅ Error handling works
✅ Can classify multiple images
✅ No console errors
✅ Network requests succeed

---

## Integration Testing

### Test Complete User Flow

**Scenario: First-time user classifies a cat image**

1. **Start Application**
   ```bash
   # Terminal 1 - Backend
   cd backend
   source venv/bin/activate
   uvicorn app:app --reload --port 8000

   # Terminal 2 - Frontend
   cd frontend
   npm start
   ```

2. **User Journey**
   - [ ] User opens `http://localhost:3000`
   - [ ] Sees welcome message and instructions
   - [ ] Backend status shows "Online"
   - [ ] User clicks upload area
   - [ ] Selects a cat image
   - [ ] Preview shows the cat image
   - [ ] User clicks "Classify Image"
   - [ ] Loading spinner appears
   - [ ] After 2-5 seconds, results appear
   - [ ] Top prediction is cat-related (e.g., "tabby cat")
   - [ ] Confidence score is reasonable (>50%)
   - [ ] User clicks "Classify Another Image"
   - [ ] Form resets, ready for new image

3. **Verify**
   - ✅ Entire flow works smoothly
   - ✅ No errors at any step
   - ✅ Predictions are accurate
   - ✅ UI is responsive and intuitive

### Test Different Image Types

Try classifying:

1. **Animals**
   - Cat → Should predict cat breeds
   - Dog → Should predict dog breeds
   - Bird → Should predict bird species

2. **Objects**
   - Car → Should predict vehicle type
   - Coffee cup → Should predict cup/mug
   - Laptop → Should predict notebook/laptop

3. **Edge Cases**
   - Very small image (< 100KB)
   - Large image (5-10MB)
   - Different formats (JPG, PNG)
   - Image with multiple objects
   - Unclear/blurry image

### Performance Testing

1. **Load Time**
   - ✅ Frontend loads in < 3 seconds
   - ✅ Backend responds to health check in < 1 second

2. **Classification Time**
   - ✅ Small image (< 500KB): 1-3 seconds
   - ✅ Medium image (500KB-2MB): 2-4 seconds
   - ✅ Large image (2-10MB): 3-6 seconds

3. **Multiple Requests**
   - ✅ Can classify 5 images in a row
   - ✅ No memory leaks
   - ✅ Performance doesn't degrade

---

## Troubleshooting

### Backend Issues

#### Issue: "ModuleNotFoundError: No module named 'fastapi'"

**Solution:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

#### Issue: "Model download timeout"

**Solution:**
```bash
# The model downloads on first run
# This is normal and may take 2-5 minutes
# Be patient and wait for download to complete

# If it keeps failing:
# 1. Check your internet connection
# 2. Try downloading manually:
python model.py
```

#### Issue: "Port 8000 already in use"

**Solution:**
```bash
# Option 1: Use a different port
uvicorn app:app --reload --port 8001

# Option 2: Find and kill process using port 8000
# macOS/Linux:
lsof -ti:8000 | xargs kill -9

# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

#### Issue: "CORS error in browser"

**Solution:**
- Check that backend is running
- Verify CORS middleware in app.py
- Try restarting backend server

### Frontend Issues

#### Issue: "npm install fails"

**Solution:**
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and package-lock.json
rm -rf node_modules package-lock.json

# Reinstall
npm install
```

#### Issue: "Module not found" errors

**Solution:**
```bash
# Make sure you're in the frontend directory
cd frontend

# Reinstall dependencies
npm install
```

#### Issue: "Cannot connect to backend"

**Solution:**
1. Check backend is running on port 8000
2. Check `frontend/src/services/api.js`:
   ```javascript
   const API_URL = 'http://localhost:8000';
   ```
3. Check browser console for errors
4. Try accessing `http://localhost:8000/health` directly

#### Issue: "Image upload fails"

**Solution:**
- Check file size (< 10MB)
- Check file type (must be image)
- Check backend logs for errors
- Verify backend `/predict` endpoint works (test with curl)

---

## Common Issues

### 1. Backend and Frontend on Same Port

**Problem:** Both trying to use same port
**Solution:** Backend on 8000, Frontend on 3000 (default)

### 2. Environment Variables Not Loading

**Problem:** API_URL not set correctly
**Solution:** Create `.env` file in frontend:
```
REACT_APP_API_URL=http://localhost:8000
```

### 3. Images Not Displaying

**Problem:** CORS or incorrect image handling
**Solution:**
- Check CORS in backend
- Verify image format (JPG, PNG)
- Check browser console

### 4. Slow Classification

**Problem:** Takes > 10 seconds to classify
**Solution:**
- Normal for first request (model initialization)
- Subsequent requests should be faster
- Check CPU usage
- Consider using GPU if available

### 5. Memory Issues

**Problem:** Application crashes or slows down
**Solution:**
- Restart both servers
- Check available memory
- Close other applications
- Limit image size

---

## Success Criteria

Before proceeding to deployment, ensure:

✅ **Backend**
- All endpoints respond correctly
- Model loads without errors
- Can classify images accurately
- No errors in server logs

✅ **Frontend**
- Application loads properly
- Can select and preview images
- Can upload and classify
- Results display correctly
- Error handling works

✅ **Integration**
- Frontend and backend communicate
- Complete user flow works
- Multiple classifications work
- Different image types work

---

## Next Steps

Once all tests pass:

1. **Review Code**
   - Read through all files
   - Understand how everything works
   - Make any desired customizations

2. **Commit to Git**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: React FastAPI image classifier"
   ```

3. **Prepare for Deployment**
   - Review DEPLOYMENT_GUIDE.md
   - Create Render account
   - Push code to GitHub

4. **Deploy**
   - Follow deployment guide
   - Test deployed application
   - Share with others!

---

## Getting Help

If you encounter issues:

1. **Check Logs**
   - Backend: Terminal where uvicorn is running
   - Frontend: Browser console (F12)

2. **Search Errors**
   - Copy error message
   - Search on Google, Stack Overflow

3. **Ask for Help**
   - Instructor or classmates
   - Include error messages and steps to reproduce

4. **Debug Systematically**
   - Test one component at a time
   - Isolate the problem
   - Check each step in the flow

---

**Happy Testing! 🎉**
