# Local Testing Guide

Complete guide for testing your React/FastAPI deep learning application locally before deployment.

## Table of Contents
1. [Setup Backend](#setup-backend)
2. [Setup Frontend](#setup-frontend)
3. [Testing the Application](#testing-the-application)
4. [Troubleshooting](#troubleshooting)
5. [Advanced Testing](#advanced-testing)

---

## Setup Backend

### Step 1: Install Python and Dependencies

**Verify Python Installation:**
```bash
python --version
# Should be 3.8 or higher
```

If Python is not installed:
- **macOS**: `brew install python`
- **Windows**: Download from https://python.org
- **Linux**: `sudo apt-get install python3 python3-pip`

### Step 2: Navigate to Backend Directory

```bash
cd backend
```

### Step 3: Create Virtual Environment

**macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Your prompt should now show (venv)
```

**Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Your prompt should now show (venv)
```

### Step 4: Install Dependencies

```bash
# Ensure pip is up to date
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

**Expected Installation Time**: 5-10 minutes (TensorFlow is large)

**Common Installation Issues:**

**Issue**: TensorFlow installation fails
```bash
# Solution: Install CPU-only version (faster, smaller)
pip install tensorflow-cpu==2.15.0
```

**Issue**: Permission denied
```bash
# Solution: Use --user flag
pip install --user -r requirements.txt
```

### Step 5: Create Environment File (Optional)

```bash
# Copy example environment file
cp .env.example .env

# Edit if needed (optional for local testing)
# nano .env  # or use your preferred editor
```

### Step 6: Start Backend Server

```bash
# Start the FastAPI server
python main.py

# Alternative method:
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Expected Output:**
```
INFO:     Will watch for changes in these directories: ['/path/to/backend']
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using StatReload
Loading MobileNetV2 model...
Model loaded successfully
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### Step 7: Verify Backend is Running

**Open a new terminal** and test:

```bash
# Test health endpoint
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","model_loaded":true}

# Test root endpoint
curl http://localhost:8000/

# Expected response:
# {"message":"Welcome to Deep Learning API","status":"active",...}
```

**Open in Browser:**
- API Documentation: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc

---

## Setup Frontend

### Step 1: Install Node.js

**Verify Node.js Installation:**
```bash
node --version
# Should be v16 or higher

npm --version
# Should be 8 or higher
```

If Node.js is not installed:
- Download from https://nodejs.org (LTS version recommended)
- Or use nvm: https://github.com/nvm-sh/nvm

### Step 2: Navigate to Frontend Directory

**Open a NEW terminal** (keep backend running):

```bash
cd frontend
```

### Step 3: Install Dependencies

```bash
# Install all packages
npm install

# Or use yarn:
# yarn install
```

**Expected Installation Time**: 2-5 minutes

**Common Installation Issues:**

**Issue**: `npm ERR! code EACCES`
```bash
# Solution: Fix npm permissions
sudo chown -R $USER ~/.npm
```

**Issue**: Package version conflicts
```bash
# Solution: Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Step 4: Create Environment File

```bash
# Copy example environment file
cp .env.example .env

# Verify API URL is correct (default is localhost:8000)
cat .env
```

**.env file should contain:**
```
REACT_APP_API_URL=http://localhost:8000
```

### Step 5: Start Frontend Development Server

```bash
# Start React development server
npm start
```

**Expected Output:**
```
Compiled successfully!

You can now view dl-frontend in the browser.

  Local:            http://localhost:3000
  On Your Network:  http://192.168.1.x:3000

Note that the development build is not optimized.
To create a production build, use npm run build.

webpack compiled successfully
```

**Browser should automatically open** to http://localhost:3000

---

## Testing the Application

### Manual Testing Workflow

#### Test 1: Check API Connection

1. Open http://localhost:3000
2. Look for "API Connected" status in the header
3. If you see "API Disconnected":
   - Verify backend is running on port 8000
   - Check browser console for errors (F12)

#### Test 2: Upload and Classify Image

1. **Prepare Test Images**:
   - Find sample images (dog, cat, car, etc.)
   - Supported formats: JPEG, PNG, WebP
   - Keep files under 10MB

2. **Upload Image**:
   - Click "Choose Image"
   - Select a test image
   - Image preview should appear

3. **Classify Image**:
   - Click "Classify Image"
   - Wait for processing (usually 2-5 seconds)
   - View predictions

4. **Verify Results**:
   - Top 5 predictions should appear
   - Confidence scores should be shown
   - Top prediction should be highlighted

#### Test 3: Test Different Images

Try various image types:
- **Animals**: Dog, cat, bird, fish
- **Objects**: Car, computer, phone, furniture
- **Food**: Pizza, banana, coffee
- **Nature**: Trees, flowers, landscapes

#### Test 4: Test Error Handling

1. **Invalid File Type**:
   - Try uploading a .txt or .pdf file
   - Should show error: "File must be an image"

2. **Large File**:
   - Try uploading a file >10MB
   - Should show error about file size

3. **API Disconnection**:
   - Stop the backend (Ctrl+C)
   - Try to classify an image
   - Should show connection error
   - Restart backend and retry

### Automated Testing

#### Backend API Testing

**Test with cURL:**

```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info

# Upload and classify image
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/image.jpg"
```

**Test with Python:**

Create `test_api.py`:

```python
import requests

# Test health endpoint
response = requests.get('http://localhost:8000/health')
print('Health:', response.json())

# Test prediction
with open('test_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict', files=files)
    print('Prediction:', response.json())
```

Run:
```bash
python test_api.py
```

#### Frontend Testing

**Test Build:**

```bash
# Create production build
npm run build

# Build should complete without errors
# Check build/ directory is created
ls build/
```

**Test with Different Browsers:**

Test on:
- Chrome/Edge
- Firefox
- Safari (if on macOS)

---

## Troubleshooting

### Backend Issues

#### Issue: "Address already in use"

**Problem**: Port 8000 is already in use

**Solution**:
```bash
# Find process using port 8000
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Kill the process
kill -9 <PID>  # macOS/Linux
taskkill /PID <PID> /F  # Windows

# Or use a different port
uvicorn main:app --reload --port 8001
```

#### Issue: "Model loading failed"

**Problem**: TensorFlow installation issue

**Solution**:
```bash
# Reinstall TensorFlow
pip uninstall tensorflow
pip install tensorflow==2.15.0

# Or use CPU-only version
pip install tensorflow-cpu==2.15.0
```

#### Issue: "Module not found"

**Problem**: Missing dependencies

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate  # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Frontend Issues

#### Issue: "Cannot connect to server"

**Problem**: Backend not running or wrong URL

**Solution**:
1. Verify backend is running: `curl http://localhost:8000/health`
2. Check `.env` file has correct API URL
3. Restart frontend: `npm start`

#### Issue: "CORS error"

**Problem**: CORS middleware not configured

**Solution**:
1. Verify `backend/main.py` includes `http://localhost:3000` in CORS origins
2. Restart backend server

#### Issue: "Module not found" in frontend

**Problem**: Missing npm packages

**Solution**:
```bash
# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

#### Issue: "Failed to compile"

**Problem**: Syntax error or missing import

**Solution**:
1. Read error message carefully
2. Check file mentioned in error
3. Verify all imports are correct
4. Check for typos

### Network Issues

#### Issue: Can't access from other devices

**Problem**: Server bound to localhost only

**Solution**:

**Backend**:
```bash
# Start with 0.0.0.0 to accept external connections
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend**:
```bash
# Set HOST environment variable
HOST=0.0.0.0 npm start
```

Then access from other devices using your IP:
```
http://192.168.1.x:3000  # Find your IP with `ifconfig` or `ipconfig`
```

---

## Advanced Testing

### Performance Testing

**Test Response Time:**

```bash
# Time a prediction request
time curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

**Load Testing** (install Apache Bench):

```bash
# Install
# macOS: brew install httpd
# Linux: sudo apt-get install apache2-utils

# Simple load test (10 requests, 2 concurrent)
ab -n 10 -c 2 http://localhost:8000/health
```

### Browser Developer Tools

**Test Network Requests:**

1. Open browser DevTools (F12)
2. Go to Network tab
3. Upload and classify an image
4. Inspect:
   - Request headers
   - Response time
   - Response size
   - Status codes

**Test Console Logs:**

1. Go to Console tab
2. Look for any errors or warnings
3. Check API connection status

### Memory Usage

**Monitor Backend Memory:**

```bash
# macOS/Linux
top -p $(pgrep -f uvicorn)

# Or use htop
htop
```

**Monitor Frontend Memory:**

- Use browser Task Manager (Chrome: Shift+Esc)
- Check memory usage of tabs

---

## Testing Checklist

Before deployment, verify:

**Backend:**
- [ ] Server starts without errors
- [ ] `/health` endpoint returns healthy status
- [ ] `/docs` shows API documentation
- [ ] `/predict` accepts and processes images
- [ ] Error handling works (invalid files, etc.)
- [ ] Model loads successfully
- [ ] CORS is configured correctly

**Frontend:**
- [ ] App loads at localhost:3000
- [ ] API status shows "Connected"
- [ ] File upload works
- [ ] Image preview displays
- [ ] Classification returns results
- [ ] Results display correctly
- [ ] Error messages show for invalid inputs
- [ ] Build completes successfully (`npm run build`)

**Integration:**
- [ ] Frontend can communicate with backend
- [ ] Images upload successfully
- [ ] Predictions return in reasonable time (<10s)
- [ ] All buttons and interactions work
- [ ] Tested on multiple browsers
- [ ] Tested with various image types

---

## Next Steps

Once local testing is complete:

1. **Code Review**: Review all code for best practices
2. **Security Check**: Ensure no API keys in code
3. **Git Commit**: Commit all changes
4. **Documentation**: Update README if needed
5. **Deploy**: Follow DEPLOYMENT_GUIDE.md

---

## Quick Reference Commands

**Backend:**
```bash
cd backend
source venv/bin/activate  # macOS/Linux
python main.py
```

**Frontend:**
```bash
cd frontend
npm start
```

**Test:**
```bash
curl http://localhost:8000/health
curl http://localhost:8000/
```

**Stop Servers:**
- Press `Ctrl+C` in each terminal

---

Happy Testing! 🧪
