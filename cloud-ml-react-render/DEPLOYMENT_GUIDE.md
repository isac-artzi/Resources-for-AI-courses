# 🚀 Deployment Guide - Render.com

Step-by-step guide for deploying your React + FastAPI application to Render.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Prepare Your Code](#prepare-your-code)
4. [Deploy Backend to Render](#deploy-backend-to-render)
5. [Deploy Frontend to Render](#deploy-frontend-to-render)
6. [Connect Frontend and Backend](#connect-frontend-and-backend)
7. [Verify Deployment](#verify-deployment)
8. [Troubleshooting](#troubleshooting)
9. [Maintenance](#maintenance)

---

## Overview

### What is Render?

Render is a cloud platform that makes deploying applications easy:
- **Free tier available** (perfect for student projects)
- **Automatic deployments** from Git
- **Built-in SSL/HTTPS**
- **Easy to use dashboard**
- **Good for learning cloud deployment**

### Architecture

```
User's Browser
     ↓
Frontend (Static Site on Render)
     ↓
Backend API (Web Service on Render)
     ↓
PyTorch Model
```

### Deployment Strategy

1. Deploy backend first (API service)
2. Get backend URL
3. Configure frontend with backend URL
4. Deploy frontend

---

## Prerequisites

### Required Accounts

1. **GitHub Account**
   - Sign up: https://github.com/join
   - Free account is sufficient

2. **Render Account**
   - Sign up: https://render.com/register
   - Free account is sufficient
   - Can sign up with GitHub (recommended)

### Prepare Locally

✅ Complete local testing (see TESTING_GUIDE.md)
✅ Ensure everything works locally
✅ Have test images ready

---

## Prepare Your Code

### Step 1: Initialize Git Repository

```bash
# Navigate to project root
cd cloud-ml-react-render

# Initialize Git repository (if not already done)
git init

# Check status
git status
```

### Step 2: Create .gitignore (if needed)

The `.gitignore` file is already created. Verify it exists:

```bash
cat .gitignore
```

### Step 3: Create Render Configuration

Create `backend/render.yaml` (already exists):

```yaml
services:
  - type: web
    name: image-classification-backend
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app:app --host 0.0.0.0 --port $PORT
```

### Step 4: Update Frontend API Configuration

Create `frontend/.env.production` (for production builds):

```bash
cd frontend

# Create .env.production file
cat > .env.production << 'EOF'
# This will be updated after backend deployment
REACT_APP_API_URL=YOUR_BACKEND_URL_HERE
EOF
```

### Step 5: Commit All Changes

```bash
# From project root
cd ..

# Add all files
git add .

# Commit
git commit -m "Initial commit: React + FastAPI image classifier"

# Verify commit
git log
```

### Step 6: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `image-classifier-app` (or your choice)
3. Description: "AI image classification app with React and FastAPI"
4. Choose **Public** (required for Render free tier)
5. Do NOT initialize with README (we already have code)
6. Click "Create repository"

### Step 7: Push to GitHub

```bash
# Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Verify remote
git remote -v

# Push code
git branch -M main
git push -u origin main

# Verify: Refresh GitHub page to see your code
```

---

## Deploy Backend to Render

### Step 1: Create Web Service

1. Go to https://dashboard.render.com/
2. Click "New +" button (top right)
3. Select "Web Service"

### Step 2: Connect GitHub Repository

1. Click "Connect account" next to GitHub
2. Authorize Render to access your repositories
3. Find your repository in the list
4. Click "Connect"

### Step 3: Configure Backend Service

Fill in the configuration:

#### Basic Settings

- **Name**: `image-classification-backend`
  - This will be part of your URL
  - Use lowercase, hyphens only

- **Region**: Choose closest to you
  - US options: Oregon, Ohio
  - Europe options: Frankfurt
  - Singapore for Asia

- **Branch**: `main`
  - Or `master` if that's your default branch

- **Root Directory**: `backend`
  - IMPORTANT: This tells Render where the backend code is

#### Build Settings

- **Runtime**: `Python 3`
  - Render will auto-detect Python

- **Build Command**:
  ```bash
  pip install -r requirements.txt
  ```

- **Start Command**:
  ```bash
  uvicorn app:app --host 0.0.0.0 --port $PORT
  ```
  - `$PORT` is automatically provided by Render

#### Advanced Settings

Click "Advanced" to expand:

- **Auto-Deploy**: `Yes`
  - Automatically deploy when you push to GitHub

- **Environment Variables**: (if needed)
  - Key: `PYTHON_VERSION`
  - Value: `3.9.0`

### Step 4: Choose Plan

- Select **Free** plan
  - 512 MB RAM
  - Shared CPU
  - Services sleep after 15 min of inactivity
  - Cold start takes ~30 seconds

### Step 5: Create Service

1. Click "Create Web Service"
2. Wait for deployment (5-10 minutes first time)

### Step 6: Monitor Deployment

Watch the logs in real-time:

**Build Phase:**
```
Installing dependencies...
Collecting fastapi
Collecting uvicorn
Collecting torch
...
Build successful!
```

**Deploy Phase:**
```
Starting service...
INFO: Started server process
INFO: Uvicorn running on http://0.0.0.0:10000
```

### Step 7: Get Backend URL

Once deployed:
1. You'll see "Your service is live 🎉"
2. Note the URL: `https://image-classification-backend.onrender.com`
3. Copy this URL (you'll need it for frontend)

### Step 8: Test Backend

Open your backend URL in browser:
```
https://YOUR-BACKEND-NAME.onrender.com
```

Should see:
```json
{
  "message": "Welcome to the Image Classification API!",
  "status": "running",
  ...
}
```

Test health endpoint:
```
https://YOUR-BACKEND-NAME.onrender.com/health
```

Test API docs:
```
https://YOUR-BACKEND-NAME.onrender.com/docs
```

### Backend Deployment Checklist

✅ Service deployed successfully
✅ Build logs show no errors
✅ Service is running (not "Deploy failed")
✅ Can access root endpoint
✅ Health check returns healthy
✅ API docs are accessible
✅ Can upload and classify image via docs

---

## Deploy Frontend to Render

### Step 1: Update Frontend Configuration

**IMPORTANT:** Update the API URL before deploying frontend

```bash
# On your local machine
cd frontend

# Edit .env.production
nano .env.production
# or
code .env.production

# Set your backend URL
REACT_APP_API_URL=https://YOUR-BACKEND-NAME.onrender.com

# Save and exit
```

### Step 2: Update src/services/api.js (Alternative)

Or directly update the code:

```javascript
// frontend/src/services/api.js
const API_URL = process.env.REACT_APP_API_URL || 'https://YOUR-BACKEND-NAME.onrender.com';
```

### Step 3: Commit and Push Changes

```bash
# From project root
cd ..

# Add changes
git add .

# Commit
git commit -m "Update API URL for production"

# Push
git push origin main
```

### Step 4: Create Static Site on Render

1. Go to Render Dashboard
2. Click "New +" → "Static Site"
3. Select your repository
4. Click "Connect"

### Step 5: Configure Frontend Service

#### Basic Settings

- **Name**: `image-classification-frontend`
  - Or any name you prefer

- **Branch**: `main`

- **Root Directory**: `frontend`
  - IMPORTANT: Points to frontend folder

#### Build Settings

- **Build Command**:
  ```bash
  npm install && npm run build
  ```
  - Installs dependencies and creates production build

- **Publish Directory**: `build`
  - Where the built files are located

#### Environment Variables

Add if using .env:
- Key: `REACT_APP_API_URL`
- Value: `https://YOUR-BACKEND-NAME.onrender.com`

### Step 6: Choose Plan

- Select **Free** plan
  - Perfect for static sites
  - Unlimited bandwidth
  - Always online (doesn't sleep)

### Step 7: Create Static Site

1. Click "Create Static Site"
2. Wait for deployment (3-5 minutes)

### Step 8: Monitor Build

Watch the logs:

```
Installing dependencies...
npm install

Building React app...
npm run build

Creating optimized production build...
Build completed!

Deploying to Render...
Deploy successful!
```

### Step 9: Get Frontend URL

Once deployed:
1. Note the URL: `https://image-classification-frontend.onrender.com`
2. Click on it to open your app

### Frontend Deployment Checklist

✅ Build completed successfully
✅ No errors in build logs
✅ Site is live
✅ Can access the URL
✅ Page loads correctly
✅ Backend status shows "Online"

---

## Connect Frontend and Backend

### Verify Connection

1. **Open Frontend URL**
   ```
   https://YOUR-FRONTEND-NAME.onrender.com
   ```

2. **Check Backend Status Badge**
   - Should show "Backend Online" (green)
   - If "Backend Offline" (red), there's a problem

3. **Test Image Upload**
   - Select an image
   - Click "Classify Image"
   - Should get predictions

### If Connection Fails

#### Check 1: Verify API URL

Open browser console (F12):
```javascript
// Should log correct backend URL
API URL: https://YOUR-BACKEND-NAME.onrender.com
```

#### Check 2: Test Backend Directly

Open backend URL:
```
https://YOUR-BACKEND-NAME.onrender.com/health
```

Should return:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### Check 3: Check CORS

Backend should allow requests from frontend:
- Open `backend/app.py`
- Find `CORSMiddleware`
- Should have `allow_origins=["*"]` (allows all origins)

For production, update to specific origin:
```python
allow_origins=[
    "https://YOUR-FRONTEND-NAME.onrender.com"
]
```

#### Check 4: Rebuild Frontend

If you changed API URL:
1. Go to Render Dashboard
2. Click on frontend service
3. Click "Manual Deploy" → "Clear build cache & deploy"

---

## Verify Deployment

### Complete End-to-End Test

1. **Open Frontend**
   - Visit your frontend URL
   - ✅ Page loads without errors

2. **Check UI**
   - ✅ Title displays correctly
   - ✅ Upload area visible
   - ✅ Backend status: "Online"

3. **Upload Image**
   - ✅ Select image
   - ✅ Preview shows
   - ✅ Click "Classify Image"
   - ✅ Loading spinner appears

4. **View Results**
   - ✅ Predictions appear (2-30 seconds)
     - First request may take 30 seconds (cold start)
     - Subsequent requests faster
   - ✅ Predictions are reasonable
   - ✅ Confidence scores display

5. **Try Multiple Images**
   - ✅ Can classify different images
   - ✅ No errors after multiple requests

### Performance Check

- **Cold Start**: First request after 15 min: 20-30 seconds
- **Warm Start**: Subsequent requests: 2-5 seconds
- **Page Load**: Should load in < 3 seconds

### Success Criteria

✅ Both services deployed successfully
✅ Frontend loads correctly
✅ Backend is reachable
✅ Can classify images
✅ Predictions are accurate
✅ No errors in console
✅ Mobile responsive (test on phone)
✅ HTTPS works (secure connection)

---

## Troubleshooting

### Issue: Backend Build Fails

**Symptoms:**
- Build logs show errors
- Service won't start

**Solutions:**

1. **Check requirements.txt**
   ```bash
   # Locally, verify all packages install
   pip install -r backend/requirements.txt
   ```

2. **Check Python version**
   - Render uses Python 3.7 by default
   - Set environment variable: `PYTHON_VERSION=3.9.0`

3. **Check for missing dependencies**
   - All imports in code must be in requirements.txt

### Issue: Backend Crashes on Start

**Symptoms:**
- Build succeeds
- Service fails to start

**Solutions:**

1. **Check start command**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port $PORT
   ```

2. **Check logs for errors**
   - Look for Python errors
   - Common: Module import errors, missing files

3. **Test locally**
   ```bash
   PORT=10000 uvicorn app:app --host 0.0.0.0 --port $PORT
   ```

### Issue: Model Download Timeout

**Symptoms:**
- Build times out
- "Download timeout" in logs

**Solutions:**

1. **Increase build timeout** (paid plans only)
2. **Pre-download model** (advanced):
   ```python
   # Add to requirements or build script
   torch.hub.load_state_dict_from_url(MODEL_URL)
   ```

### Issue: Frontend Can't Reach Backend

**Symptoms:**
- Backend status: "Offline"
- Upload fails with network error

**Solutions:**

1. **Check API URL in frontend**
   ```bash
   # Must be HTTPS backend URL
   REACT_APP_API_URL=https://YOUR-BACKEND.onrender.com
   ```

2. **Rebuild frontend after changing URL**
   - Manual Deploy → Clear cache & deploy

3. **Check CORS settings in backend**
   - Must allow frontend origin

4. **Test backend directly**
   - Open backend URL in browser
   - Should see welcome message

### Issue: Cold Start Too Slow

**Symptoms:**
- First request after inactivity takes 30+ seconds

**Explanation:**
- Free tier services sleep after 15 min
- Cold start includes:
  - Spinning up container
  - Loading Python
  - Loading PyTorch model

**Solutions:**
- **Upgrade to paid plan** ($7/month keeps service always on)
- **Use a keep-alive service** (ping endpoint every 10 min)
- **Accept the limitation** (free tier trade-off)

### Issue: Out of Memory

**Symptoms:**
- Service crashes
- "Out of memory" in logs

**Solutions:**
- Free tier: 512MB RAM
- PyTorch model uses ~300-400MB
- Solutions:
  1. Use smaller model (MobileNet)
  2. Upgrade to paid plan (more RAM)
  3. Reduce batch size

### Issue: Environment Variables Not Working

**Symptoms:**
- API URL not updated
- Configuration not applied

**Solutions:**

1. **Check variable name**
   - React: Must start with `REACT_APP_`
   - Correct: `REACT_APP_API_URL`
   - Wrong: `API_URL`

2. **Rebuild after adding variables**
   - Environment variables are embedded during build

3. **Check .env file**
   - Must be `.env.production` for production builds

---

## Maintenance

### Update the Application

```bash
# Make changes locally
# Test changes locally
# Commit changes
git add .
git commit -m "Description of changes"

# Push to GitHub
git push origin main

# Render auto-deploys (if enabled)
# Or manually deploy from dashboard
```

### Monitor Services

1. **Check Status**
   - Dashboard shows service status
   - Green: Running
   - Yellow: Deploying
   - Red: Failed

2. **View Logs**
   - Click on service
   - "Logs" tab shows real-time logs

3. **Check Metrics** (paid plans)
   - CPU usage
   - Memory usage
   - Request count

### Manage Costs

**Free Tier Limits:**
- 750 hours/month web services
- Unlimited static sites
- Unlimited bandwidth

**When to Upgrade:**
- Need faster cold starts
- Need more memory
- Need persistent service
- Student project: Stay on free tier!

### Custom Domain (Optional)

1. Buy domain (Namecheap, Google Domains)
2. In Render:
   - Settings → Custom Domains
   - Add domain
   - Update DNS records
   - Wait for SSL certificate

---

## Next Steps

### Share Your Project

1. **Get URLs**
   - Frontend: `https://YOUR-FRONTEND.onrender.com`
   - Backend: `https://YOUR-BACKEND.onrender.com`

2. **Share**
   - Add to resume/portfolio
   - Share with friends/classmates
   - Demo to instructor

3. **Document**
   - Add screenshots to README
   - Write about what you learned
   - Explain technical choices

### Improve Your App

Ideas for enhancements:
- Add user authentication
- Store classification history
- Support batch uploads
- Add more models
- Improve UI/UX
- Add analytics

### Learn More

- **Render Docs**: https://render.com/docs
- **FastAPI Docs**: https://fastapi.tiangolo.com
- **React Docs**: https://react.dev
- **PyTorch Tutorials**: https://pytorch.org/tutorials

---

## Conclusion

Congratulations! 🎉

You've successfully:
✅ Built a full-stack deep learning application
✅ Deployed backend to the cloud
✅ Deployed frontend to the cloud
✅ Connected frontend and backend
✅ Created a publicly accessible web app

This is a real cloud deployment that you can:
- Show to potential employers
- Share with others
- Build upon for more projects
- Use as a portfolio piece

**Keep building and learning!** 🚀

---

## Getting Help

If you get stuck:

1. **Check Render Logs**
   - Most errors visible in logs
   - Click service → Logs tab

2. **Render Community**
   - https://community.render.com
   - Active community forum

3. **Documentation**
   - Render: https://render.com/docs
   - FastAPI: https://fastapi.tiangolo.com
   - React: https://react.dev

4. **Ask Instructor/Classmates**
   - Share specific error messages
   - Include steps to reproduce

5. **Stack Overflow**
   - Tag: render, fastapi, react
   - Include error details

---

**Happy Deploying! 🌐**
