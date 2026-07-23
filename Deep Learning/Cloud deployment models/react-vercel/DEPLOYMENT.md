# Deployment Guide
## React + FastAPI Deep Learning Application

This guide provides step-by-step instructions for deploying your application to production.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Backend Deployment (Railway)](#backend-deployment-railway)
3. [Frontend Deployment (Vercel)](#frontend-deployment-vercel)
4. [Alternative Deployment Options](#alternative-deployment-options)
5. [Environment Variables](#environment-variables)
6. [Testing Deployment](#testing-deployment)
7. [Troubleshooting](#troubleshooting)
8. [Maintenance](#maintenance)

## ✅ Prerequisites

Before deploying, ensure you have:

- [x] GitHub account with your code pushed to a repository
- [x] Railway account (for backend)
- [x] Vercel account (for frontend)
- [x] Backend tested locally
- [x] Frontend tested locally
- [x] All environment variables documented

### Create GitHub Repository

```bash
# Navigate to your project root
cd react-vercel

# Initialize git (if not already done)
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: React + FastAPI Deep Learning App"

# Add remote repository (replace with your GitHub URL)
git remote add origin https://github.com/yourusername/your-repo.git

# Push to GitHub
git push -u origin main
```

## 🚂 Backend Deployment (Railway)

Railway provides a simple platform for deploying Python applications with persistent servers.

### Step 1: Prepare Backend for Deployment

1. **Verify requirements.txt**
   ```bash
   cd backend
   pip freeze > requirements.txt
   ```

2. **Create Procfile** (Optional, Railway auto-detects)
   ```bash
   echo "web: uvicorn app.main:app --host 0.0.0.0 --port \$PORT" > Procfile
   ```

3. **Verify runtime.txt**
   ```bash
   cat runtime.txt
   # Should contain: python-3.11.7
   ```

### Step 2: Deploy to Railway

#### Option A: Deploy via Railway Dashboard

1. **Go to Railway**: [https://railway.app/](https://railway.app/)

2. **Login/Sign Up** using GitHub

3. **Create New Project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository
   - Railway will detect Python automatically

4. **Configure Build Settings**
   - Root Directory: `backend`
   - Build Command: (leave empty, auto-detected)
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

5. **Add Environment Variables** (Optional)
   - Click on your service
   - Go to "Variables" tab
   - Add variables as needed:
     ```
     ENVIRONMENT=production
     LOG_LEVEL=INFO
     ```

6. **Deploy**
   - Railway will automatically build and deploy
   - Wait for deployment to complete
   - Copy the generated URL (e.g., `https://your-app.railway.app`)

#### Option B: Deploy via Railway CLI

1. **Install Railway CLI**
   ```bash
   npm i -g @railway/cli
   ```

2. **Login**
   ```bash
   railway login
   ```

3. **Initialize Project**
   ```bash
   cd backend
   railway init
   ```

4. **Deploy**
   ```bash
   railway up
   ```

5. **Get URL**
   ```bash
   railway domain
   ```

### Step 3: Verify Backend Deployment

1. **Test Health Endpoint**
   ```bash
   curl https://your-app.railway.app/health
   ```

   Expected response:
   ```json
   {
     "status": "healthy",
     "model_loaded": true,
     "timestamp": "2024-01-15T10:30:00"
   }
   ```

2. **Check API Documentation**
   - Visit: `https://your-app.railway.app/docs`
   - Should see FastAPI Swagger UI

3. **Test with Sample Image**
   ```bash
   curl -X POST "https://your-app.railway.app/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/test/image.jpg"
   ```

### Step 4: Monitor Backend

Railway provides monitoring tools:
- **Logs**: View real-time logs in the dashboard
- **Metrics**: CPU, memory, and network usage
- **Deployments**: Track deployment history

Access logs:
```bash
railway logs
```

## 🚀 Frontend Deployment (Vercel)

### Step 1: Prepare Frontend

1. **Update API URL**
   ```bash
   cd frontend
   ```

2. **Edit .env.production**
   ```env
   VITE_API_URL=https://your-app.railway.app
   ```

3. **Test Production Build Locally**
   ```bash
   npm run build
   npm run preview
   ```

4. **Commit Changes**
   ```bash
   git add .env.production
   git commit -m "Update production API URL"
   git push
   ```

### Step 2: Deploy to Vercel

#### Option A: Deploy via Vercel Dashboard

1. **Go to Vercel**: [https://vercel.com/](https://vercel.com/)

2. **Login** using GitHub

3. **Import Project**
   - Click "Add New..." > "Project"
   - Import your GitHub repository
   - Click "Import"

4. **Configure Project**
   - **Framework Preset**: Vite
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
   - **Install Command**: `npm install`

5. **Add Environment Variables**
   - Click "Environment Variables"
   - Add:
     ```
     VITE_API_URL = https://your-app.railway.app
     ```
   - Select "Production" environment

6. **Deploy**
   - Click "Deploy"
   - Wait for build to complete
   - Copy the generated URL (e.g., `https://your-app.vercel.app`)

#### Option B: Deploy via Vercel CLI

1. **Install Vercel CLI**
   ```bash
   npm i -g vercel
   ```

2. **Login**
   ```bash
   vercel login
   ```

3. **Deploy**
   ```bash
   cd frontend
   vercel
   ```

4. **Follow Prompts**
   - Set up and deploy: `Y`
   - Which scope: (select your account)
   - Link to existing project: `N`
   - Project name: (enter name)
   - Directory: `./` (current directory)
   - Want to override settings: `Y`
   - Build Command: `npm run build`
   - Output Directory: `dist`

5. **Deploy to Production**
   ```bash
   vercel --prod
   ```

### Step 3: Verify Frontend Deployment

1. **Visit Your App**
   - Open `https://your-app.vercel.app` in browser

2. **Test Complete Flow**
   - Upload an image
   - Click "Classify Image"
   - Verify predictions appear

3. **Check Browser Console**
   - Open Developer Tools (F12)
   - Look for any errors
   - Verify API calls are successful

## 🔄 Alternative Deployment Options

### Backend Alternatives

#### 1. Render

**Pros**: Free tier, easy setup, similar to Railway

1. Go to [render.com](https://render.com/)
2. Create New > Web Service
3. Connect GitHub repository
4. Configure:
   - Root Directory: `backend`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Deploy

#### 2. Google Cloud Run

**Pros**: Serverless, auto-scaling, pay-per-use

Requires Docker:
```dockerfile
# backend/Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Deploy:
```bash
gcloud run deploy --source .
```

#### 3. AWS Elastic Beanstalk

**Pros**: AWS ecosystem, scalable

Requires more setup but highly scalable.

### Frontend Alternatives

#### 1. Netlify

Similar to Vercel:
1. Connect GitHub
2. Build Command: `npm run build`
3. Publish Directory: `dist`

#### 2. GitHub Pages

For static sites (backend must be deployed elsewhere):
```bash
npm run build
npx gh-pages -d dist
```

#### 3. AWS S3 + CloudFront

Enterprise-grade static hosting.

## 🔐 Environment Variables

### Backend Environment Variables

| Variable | Example | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port (auto-set by Railway) |
| `ENVIRONMENT` | `production` | Environment name |
| `LOG_LEVEL` | `INFO` | Logging level |
| `ALLOWED_ORIGINS` | `https://your-app.vercel.app` | CORS origins |

### Frontend Environment Variables

| Variable | Example | Description |
|----------|---------|-------------|
| `VITE_API_URL` | `https://backend.railway.app` | Backend API URL |
| `VITE_APP_NAME` | `AI Classifier` | Application name |

## 🧪 Testing Deployment

### Automated Testing Script

```bash
#!/bin/bash
# test-deployment.sh

BACKEND_URL="https://your-app.railway.app"
FRONTEND_URL="https://your-app.vercel.app"

echo "Testing Backend..."
curl -f $BACKEND_URL/health || echo "❌ Backend health check failed"

echo "Testing Frontend..."
curl -f $FRONTEND_URL || echo "❌ Frontend not accessible"

echo "✅ All tests passed!"
```

Run with:
```bash
chmod +x test-deployment.sh
./test-deployment.sh
```

## 🐛 Troubleshooting

### Common Issues

#### 1. CORS Errors

**Symptom**: Frontend can't connect to backend

**Solution**: Update CORS settings in `backend/app/main.py`:
```python
allowed_origins = [
    "https://your-app.vercel.app",
    "https://*.vercel.app"  # All preview deployments
]
```

#### 2. 504 Gateway Timeout

**Symptom**: Requests timeout

**Solution**:
- Increase timeout in Railway (default 30s)
- Optimize model loading
- Use lighter model

#### 3. Environment Variables Not Loading

**Symptom**: App uses default values

**Solution**:
- Check variable names (must start with `VITE_`)
- Redeploy after adding variables
- Check Vercel deployment logs

#### 4. Build Fails

**Symptom**: Deployment fails during build

**Solution**:
- Check package.json dependencies
- Verify Node version
- Check build logs for specific errors

## 🔧 Maintenance

### Updating the Application

1. **Make Changes Locally**
   ```bash
   # Edit code
   git add .
   git commit -m "Description of changes"
   git push
   ```

2. **Auto-Deployment**
   - Both Railway and Vercel auto-deploy on push
   - Monitor deployment status in dashboards

### Monitoring

1. **Railway Dashboard**
   - View logs
   - Monitor resource usage
   - Check deployment status

2. **Vercel Dashboard**
   - View deployment history
   - Check analytics
   - Monitor build times

### Backup and Rollback

#### Vercel Rollback
```bash
vercel rollback
```

Or use dashboard to rollback to previous deployment.

#### Railway Rollback
Use dashboard to redeploy previous version.

## 📊 Performance Optimization

### Backend Optimization

1. **Use Gunicorn with Multiple Workers**
   ```bash
   gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

2. **Enable Caching**
   - Cache model in memory
   - Use Redis for prediction caching

3. **Use TensorFlow Lite**
   - Smaller model size
   - Faster inference

### Frontend Optimization

1. **Code Splitting**
   - Already handled by Vite
   - Lazy load components if needed

2. **Image Optimization**
   - Compress images before upload
   - Use WebP format

3. **CDN**
   - Vercel automatically uses CDN
   - Assets served from edge locations

## 🎓 Best Practices

1. **Use Environment Variables**: Never hardcode URLs or secrets
2. **Enable HTTPS**: Both Railway and Vercel provide SSL automatically
3. **Monitor Logs**: Regular check logs for errors
4. **Test Before Deploy**: Always test locally first
5. **Version Control**: Use git tags for releases
6. **Documentation**: Keep README updated

## 📚 Additional Resources

- [Railway Documentation](https://docs.railway.app/)
- [Vercel Documentation](https://vercel.com/docs)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Vite Production Build](https://vitejs.dev/guide/build.html)

---

**Need Help?** Check the main [README.md](README.md) or contact your instructor.
