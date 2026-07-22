# Deployment Guide: Step-by-Step

This guide provides detailed instructions for deploying the React/FastAPI deep learning app to Vercel (frontend) and Render (backend).

## Table of Contents
1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Deploy Backend to Render](#deploy-backend-to-render)
3. [Deploy Frontend to Vercel](#deploy-frontend-to-vercel)
4. [Post-Deployment Configuration](#post-deployment-configuration)
5. [Verification](#verification)

---

## Pre-Deployment Checklist

Before deploying, ensure you have:

- [ ] GitHub account created
- [ ] Render account created (https://render.com)
- [ ] Vercel account created (https://vercel.com)
- [ ] Code pushed to a GitHub repository
- [ ] All local tests passing
- [ ] Environment variables documented

---

## Deploy Backend to Render

### Step 1: Create GitHub Repository

```bash
# Initialize git repository (if not already done)
cd /path/to/project
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: FastAPI backend and React frontend"

# Create repository on GitHub (via web interface)
# Then link and push:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

### Step 2: Sign Up for Render

1. Go to https://render.com
2. Click "Get Started for Free"
3. Sign up with GitHub (recommended for easy integration)

### Step 3: Create New Web Service

1. From Render Dashboard, click **"New +"**
2. Select **"Web Service"**
3. Connect your GitHub repository
   - Click "Configure account" if needed
   - Grant Render access to your repository

### Step 4: Configure Web Service

Fill in the following settings:

**Basic Settings:**
- **Name**: `dl-api-backend` (or your preferred name)
- **Region**: Choose closest to your users (e.g., Oregon, Frankfurt)
- **Branch**: `main`
- **Root Directory**: `backend`
- **Environment**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

**Instance Type:**
- Select **"Free"** for testing (note: free tier spins down after inactivity)
- For production: **"Starter"** ($7/month) or higher

**Advanced Settings (Optional):**
- **Auto-Deploy**: `Yes` (deploys automatically on git push)
- **Health Check Path**: `/health`

### Step 5: Add Environment Variables

Click "Advanced" and add environment variables:

| Key | Value |
|-----|-------|
| `PYTHON_VERSION` | `3.9.0` |
| `ENVIRONMENT` | `production` |

### Step 6: Deploy

1. Click **"Create Web Service"**
2. Wait for deployment (5-15 minutes for first deploy)
3. Monitor logs for any errors
4. Once deployed, note your backend URL:
   ```
   https://dl-api-backend.onrender.com
   ```

### Step 7: Test Backend

Test your backend is working:

```bash
# Health check
curl https://dl-api-backend.onrender.com/health

# Root endpoint
curl https://dl-api-backend.onrender.com/

# API documentation
# Open in browser: https://dl-api-backend.onrender.com/docs
```

---

## Deploy Frontend to Vercel

### Step 1: Sign Up for Vercel

1. Go to https://vercel.com
2. Click "Sign Up"
3. Sign up with GitHub (recommended)

### Step 2: Update Frontend Configuration

Before deploying, update the API URL:

**Option A: Using Environment Variable (Recommended)**

Create `.env.production` in `frontend/`:

```bash
REACT_APP_API_URL=https://dl-api-backend.onrender.com
```

**Option B: Update config.js directly**

Edit `frontend/src/config.js`:

```javascript
export const API_URL = process.env.REACT_APP_API_URL || 'https://dl-api-backend.onrender.com';
```

Commit and push changes:

```bash
git add .
git commit -m "Update API URL for production"
git push
```

### Step 3: Import Project to Vercel

**Method 1: Vercel Dashboard (Recommended for beginners)**

1. Go to https://vercel.com/dashboard
2. Click **"Add New..."** → **"Project"**
3. Import your GitHub repository
4. Configure project:
   - **Framework Preset**: `Create React App`
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build` (auto-detected)
   - **Output Directory**: `build` (auto-detected)
   - **Install Command**: `npm install` (auto-detected)

5. Add Environment Variable:
   - Click "Environment Variables"
   - **Name**: `REACT_APP_API_URL`
   - **Value**: `https://dl-api-backend.onrender.com`
   - **Environment**: All (Production, Preview, Development)

6. Click **"Deploy"**

**Method 2: Vercel CLI**

```bash
# Install Vercel CLI
npm install -g vercel

# Navigate to frontend directory
cd frontend

# Deploy
vercel

# Follow prompts:
# - Set up and deploy? Yes
# - Which scope? Your account
# - Link to existing project? No
# - Project name? dl-frontend
# - Directory? ./
# - Override settings? No

# For production deployment:
vercel --prod
```

### Step 4: Configure Custom Domain (Optional)

1. In Vercel Dashboard, go to your project
2. Navigate to **Settings** → **Domains**
3. Add your custom domain
4. Update DNS records as instructed

### Step 5: Note Frontend URL

Your frontend will be deployed at:
```
https://dl-frontend.vercel.app
```
or your custom domain.

---

## Post-Deployment Configuration

### Update CORS in Backend

Now that you have your Vercel URL, update the backend CORS settings:

1. Edit `backend/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",                      # Local development
        "https://dl-frontend.vercel.app",             # Your Vercel production
        "https://dl-frontend-*.vercel.app",           # Vercel preview deployments
        "https://your-custom-domain.com",             # Custom domain (if applicable)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

2. Commit and push:

```bash
git add backend/main.py
git commit -m "Update CORS for production frontend URL"
git push
```

3. Render will automatically redeploy with the new CORS settings (if auto-deploy is enabled)

---

## Verification

### Test Complete Workflow

1. **Visit Frontend**: Open https://dl-frontend.vercel.app
2. **Check API Status**: Should show "API Connected"
3. **Upload Image**: Select a test image
4. **Classify**: Click "Classify Image"
5. **View Results**: Verify predictions appear correctly

### Troubleshooting Verification

If something doesn't work:

1. **Check Browser Console** (F12):
   - Look for CORS errors
   - Check API request/response
   - Verify API URL is correct

2. **Check Render Logs**:
   - Go to Render Dashboard → Your Service → Logs
   - Look for errors or failed requests

3. **Check Vercel Logs**:
   - Go to Vercel Dashboard → Your Project → Deployments
   - Click on deployment → Runtime Logs

4. **Test Backend Directly**:
   ```bash
   # Should return API info
   curl https://dl-api-backend.onrender.com/

   # Should return healthy status
   curl https://dl-api-backend.onrender.com/health
   ```

---

## Common Deployment Issues

### Issue 1: Backend Times Out on First Request

**Problem**: Render free tier "spins down" after inactivity. First request takes 30-60 seconds.

**Solutions**:
- Upgrade to Starter plan ($7/month) for always-on service
- Implement a keep-alive ping service
- Display loading message to users

### Issue 2: CORS Errors

**Problem**: Frontend can't access backend API.

**Solutions**:
- Verify Vercel URL is added to CORS `allow_origins`
- Check for typos in URLs
- Ensure HTTPS is used (not HTTP)

### Issue 3: Environment Variables Not Working

**Problem**: API_URL not being read.

**Solutions**:
- Vercel: Ensure variable starts with `REACT_APP_`
- Redeploy after adding environment variables
- Check variable is set for correct environment (Production/Preview)

### Issue 4: Model Loading Fails

**Problem**: TensorFlow model doesn't load on Render.

**Solutions**:
- Check Render logs for specific error
- Verify `requirements.txt` has correct TensorFlow version
- Consider using `tensorflow-cpu` for smaller deployment
- Increase instance type if running out of memory

### Issue 5: Build Fails on Vercel

**Problem**: Frontend build fails during deployment.

**Solutions**:
- Check Vercel build logs for specific error
- Verify `package.json` has all dependencies
- Test build locally: `npm run build`
- Check for ESLint warnings (they block builds)

---

## Monitoring and Maintenance

### Set Up Monitoring

1. **Render Monitoring**:
   - Enable email notifications for failures
   - Set up health check endpoint

2. **Vercel Analytics** (Optional):
   - Enable Vercel Analytics in dashboard
   - Monitor performance and errors

3. **External Monitoring** (Optional):
   - Use UptimeRobot (free) to ping your API
   - Keeps Render free tier from sleeping
   - Alerts you when service is down

### Regular Maintenance

- **Monitor Logs**: Check regularly for errors
- **Update Dependencies**: Keep packages up to date
- **Review Costs**: Monitor Render and Vercel usage
- **Backup**: Keep git repository updated

---

## Scaling Considerations

### When to Upgrade

Upgrade your infrastructure when:
- **Backend**: >1000 requests/day or need <1s response time
- **Frontend**: >100GB bandwidth/month or need better performance
- **Storage**: Need to save user data or model results

### Recommended Upgrades

1. **Render**:
   - Free → Starter ($7/month): Always-on, faster cold starts
   - Starter → Standard ($25/month): More CPU/RAM

2. **Vercel**:
   - Hobby (Free) → Pro ($20/month): Better performance, analytics
   - Pro → Enterprise: Custom requirements

3. **Database** (if needed):
   - Add PostgreSQL database on Render (free tier available)
   - Store predictions, user data, analytics

---

## Next Steps

After successful deployment:

1. **Custom Domain**: Add your own domain name
2. **Analytics**: Set up Google Analytics or Vercel Analytics
3. **Error Tracking**: Add Sentry for error monitoring
4. **Authentication**: Implement user authentication
5. **Database**: Add PostgreSQL for data persistence
6. **Caching**: Implement Redis for faster responses
7. **CI/CD**: Set up automated testing before deployment

---

## Resources

- **Render Documentation**: https://render.com/docs
- **Vercel Documentation**: https://vercel.com/docs
- **FastAPI Deployment**: https://fastapi.tiangolo.com/deployment/
- **React Deployment**: https://create-react-app.dev/docs/deployment/

## Support

If you encounter issues:
1. Check the troubleshooting sections
2. Review deployment logs
3. Consult official documentation
4. Search community forums (Render, Vercel, Stack Overflow)

---

**Congratulations!** Your deep learning app is now deployed and accessible worldwide! 🎉
