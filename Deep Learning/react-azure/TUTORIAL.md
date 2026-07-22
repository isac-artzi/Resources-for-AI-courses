# React + FastAPI App Deployment on Azure - Complete Tutorial

## Table of Contents
1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Project Structure](#project-structure)
4. [Backend Setup (FastAPI)](#backend-setup-fastapi)
5. [Frontend Setup (React)](#frontend-setup-react)
6. [Local Testing](#local-testing)
7. [Azure Deployment](#azure-deployment)
8. [Troubleshooting](#troubleshooting)

---

## Project Overview

This tutorial will guide you through building a full-stack Todo application with:
- **Backend**: FastAPI (Python) - RESTful API with CORS support
- **Frontend**: React (TypeScript) - Modern UI with Vite
- **Database**: In-memory (easily upgradable to PostgreSQL/MongoDB)
- **Deployment**: Azure Static Web Apps (Frontend) + Azure Container Apps (Backend) - **FREE TIER**

### What You'll Build
A Todo application where users can:
- Create new todos
- View all todos
- Mark todos as complete/incomplete
- Delete todos

---

## Prerequisites

### Required Software
- **Node.js** (v18 or higher) - [Download](https://nodejs.org/)
- **Python** (v3.9 or higher) - [Download](https://www.python.org/)
- **Git** - [Download](https://git-scm.com/)
- **Azure CLI** - [Install Instructions](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)
- **Docker** (for Azure deployment) - [Download](https://www.docker.com/)

### Azure Account
- Create a free Azure account at [https://azure.microsoft.com/free/](https://azure.microsoft.com/free/)
- You get $200 credit for 30 days + 12 months of free services

### Knowledge Requirements
- Basic Python and JavaScript/TypeScript
- Basic understanding of REST APIs
- Basic Git commands

---

## Project Structure

```
react-azure/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py            # Main FastAPI application
│   │   ├── models.py          # Data models
│   │   └── routes.py          # API routes
│   ├── requirements.txt       # Python dependencies
│   ├── Dockerfile             # Docker configuration for Azure
│   └── .dockerignore
├── frontend/                   # React frontend
│   ├── public/
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── services/          # API service layer
│   │   ├── App.tsx           # Main App component
│   │   ├── main.tsx          # Entry point
│   │   └── index.css         # Global styles
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   └── staticwebapp.config.json  # Azure Static Web Apps config
├── .gitignore
├── README.md
└── TUTORIAL.md               # This file
```

---

## Backend Setup (FastAPI)

### Step 1: Create Backend Directory Structure

```bash
mkdir -p backend/app
cd backend
```

### Step 2: Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### Step 3: Create requirements.txt

See `backend/requirements.txt` for all dependencies.

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Create FastAPI Application Files

The backend consists of several files (see the `backend/app/` directory):
- `models.py` - Pydantic models for data validation
- `routes.py` - API endpoint definitions
- `main.py` - Main application with CORS configuration

### Step 6: Run Backend Locally

```bash
# Make sure you're in the backend directory with venv activated
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` to see the interactive API documentation (Swagger UI).

**API Endpoints:**
- `GET /` - Health check
- `GET /api/todos` - Get all todos
- `POST /api/todos` - Create a new todo
- `PUT /api/todos/{id}` - Update a todo
- `DELETE /api/todos/{id}` - Delete a todo

---

## Frontend Setup (React)

### Step 1: Create React App with Vite

```bash
# From the project root directory
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
```

### Step 2: Install Additional Dependencies

```bash
npm install axios
```

### Step 3: Create Frontend Files

The frontend consists of:
- **Components**: TodoList, TodoItem, TodoForm
- **Services**: API service for backend communication
- **Styling**: Modern CSS with responsive design

All files are in the `frontend/src/` directory.

### Step 4: Configure Environment Variables

Create `frontend/.env.local`:
```env
VITE_API_URL=http://localhost:8000
```

For production (Azure), this will be updated to your Azure Container Apps URL.

### Step 5: Run Frontend Locally

```bash
cd frontend
npm run dev
```

Visit `http://localhost:5173` to see your app.

---

## Local Testing

### Full Stack Testing

1. **Start Backend** (Terminal 1):
```bash
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2. **Start Frontend** (Terminal 2):
```bash
cd frontend
npm run dev
```

3. **Test the Application**:
   - Open `http://localhost:5173`
   - Create a new todo
   - Mark todos as complete
   - Delete todos
   - Check the browser console for any errors
   - Verify API calls in Network tab

4. **Test API Directly**:
   - Visit `http://localhost:8000/docs`
   - Try each endpoint using the Swagger UI

---

## Azure Deployment

### Overview
- **Frontend**: Azure Static Web Apps (FREE tier)
- **Backend**: Azure Container Apps (FREE tier - 180,000 vCPU seconds + 360,000 GiB seconds per month)

### Step 1: Prepare Your Code for Deployment

#### A. Initialize Git Repository

```bash
# From project root
git init
git add .
git commit -m "Initial commit: React + FastAPI Todo App"
```

#### B. Create GitHub Repository

1. Go to [GitHub](https://github.com)
2. Click "New Repository"
3. Name it `react-fastapi-azure-todo`
4. Don't initialize with README (we already have code)
5. Click "Create Repository"

```bash
# Link local repo to GitHub
git remote add origin https://github.com/YOUR_USERNAME/react-fastapi-azure-todo.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy Backend to Azure Container Apps

#### A. Login to Azure

```bash
az login
```

#### B. Create Resource Group

```bash
# Set variables
RESOURCE_GROUP="rg-todo-app"
LOCATION="eastus"
CONTAINER_APP_NAME="todo-api"
CONTAINER_REGISTRY_NAME="todoappreg"  # Must be globally unique, use lowercase letters and numbers only

# Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION
```

#### C. Create Azure Container Registry (ACR)

```bash
# Create container registry
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $CONTAINER_REGISTRY_NAME \
  --sku Basic \
  --admin-enabled true

# Login to ACR
az acr login --name $CONTAINER_REGISTRY_NAME
```

#### D. Build and Push Docker Image

```bash
# Get ACR login server
ACR_LOGIN_SERVER=$(az acr show --name $CONTAINER_REGISTRY_NAME --query loginServer --output tsv)

# Build and push Docker image
cd backend
docker build -t $ACR_LOGIN_SERVER/todo-api:latest .
docker push $ACR_LOGIN_SERVER/todo-api:latest
```

#### E. Create Container Apps Environment

```bash
# Create environment
az containerapp env create \
  --name todo-app-env \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION
```

#### F. Deploy Container App

```bash
# Get ACR credentials
ACR_USERNAME=$(az acr credential show --name $CONTAINER_REGISTRY_NAME --query username --output tsv)
ACR_PASSWORD=$(az acr credential show --name $CONTAINER_REGISTRY_NAME --query passwords[0].value --output tsv)

# Create container app
az containerapp create \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --environment todo-app-env \
  --image $ACR_LOGIN_SERVER/todo-api:latest \
  --registry-server $ACR_LOGIN_SERVER \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --target-port 8000 \
  --ingress external \
  --min-replicas 0 \
  --max-replicas 1

# Get the backend URL
BACKEND_URL=$(az containerapp show \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --query properties.configuration.ingress.fqdn \
  --output tsv)

echo "Backend URL: https://$BACKEND_URL"
```

**Note the backend URL** - you'll need it for the frontend configuration!

### Step 3: Deploy Frontend to Azure Static Web Apps

#### A. Update Frontend Configuration

1. Update `frontend/.env.production`:
```env
VITE_API_URL=https://YOUR_BACKEND_URL
```
Replace `YOUR_BACKEND_URL` with the URL from Step 2F.

2. Update `frontend/staticwebapp.config.json` if needed (already configured).

#### B. Build Frontend

```bash
cd frontend
npm run build
```

#### C. Deploy to Azure Static Web Apps via GitHub

1. **Via Azure Portal** (Easiest):
   - Go to [Azure Portal](https://portal.azure.com)
   - Click "Create a resource"
   - Search for "Static Web App"
   - Click "Create"
   - Fill in:
     - Resource Group: Select `rg-todo-app`
     - Name: `todo-frontend`
     - Plan: Free
     - Region: East US 2 (or closest to you)
     - Source: GitHub
     - Sign in to GitHub and authorize Azure
     - Organization: Your GitHub username
     - Repository: `react-fastapi-azure-todo`
     - Branch: `main`
     - Build Presets: React
     - App location: `/frontend`
     - Api location: (leave empty)
     - Output location: `dist`
   - Click "Review + Create"
   - Click "Create"

2. **Via Azure CLI** (Alternative):

```bash
# Install Static Web Apps CLI
npm install -g @azure/static-web-apps-cli

# Create Static Web App
az staticwebapp create \
  --name todo-frontend \
  --resource-group $RESOURCE_GROUP \
  --source https://github.com/YOUR_USERNAME/react-fastapi-azure-todo \
  --location "East US 2" \
  --branch main \
  --app-location "frontend" \
  --output-location "dist" \
  --login-with-github
```

#### D. Configure GitHub Action

Azure automatically creates a GitHub Action workflow. To update environment variables:

1. Go to your GitHub repository
2. Click "Settings" → "Secrets and variables" → "Actions"
3. Add a new repository secret:
   - Name: `VITE_API_URL`
   - Value: `https://YOUR_BACKEND_URL`

4. Update `.github/workflows/azure-static-web-apps-*.yml`:

```yaml
# Add under 'env:' section
env:
  VITE_API_URL: ${{ secrets.VITE_API_URL }}
```

5. Commit and push changes to trigger redeployment.

### Step 4: Verify Deployment

1. **Check Backend**:
   ```bash
   curl https://YOUR_BACKEND_URL/api/todos
   ```

2. **Check Frontend**:
   - Go to your Static Web App URL (find it in Azure Portal)
   - Test creating, updating, and deleting todos
   - Open browser DevTools → Network tab to verify API calls

### Step 5: Update CORS Settings (If Needed)

If you get CORS errors, update `backend/app/main.py`:

```python
# Update origins to include your Static Web App URL
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://YOUR_STATIC_WEB_APP_URL.azurestaticapps.net",
]
```

Rebuild and redeploy the backend:

```bash
cd backend
docker build -t $ACR_LOGIN_SERVER/todo-api:latest .
docker push $ACR_LOGIN_SERVER/todo-api:latest

# Update container app
az containerapp update \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --image $ACR_LOGIN_SERVER/todo-api:latest
```

---

## Cost Management (Staying Free)

### Free Tier Limits

**Azure Static Web Apps (Free)**:
- 100 GB bandwidth per subscription per month
- Custom domains supported
- Perfect for this project

**Azure Container Apps (Free)**:
- 180,000 vCPU-seconds per month
- 360,000 GiB-seconds per month
- With `--min-replicas 0`, scales to zero when not in use
- Generous for low-traffic apps

**Azure Container Registry (Basic)**:
- $0.167 per day (~$5/month)
- 10 GB storage
- This is the only paid service, but minimal cost

### Tips to Stay Free/Low-Cost

1. **Scale to Zero**: Container Apps are configured with `--min-replicas 0`
2. **Delete When Not Needed**:
   ```bash
   az group delete --name $RESOURCE_GROUP --yes
   ```
3. **Monitor Usage**:
   - Check Azure Portal → Cost Management
   - Set up billing alerts

---

## Troubleshooting

### Common Issues

#### 1. CORS Errors
**Problem**: Frontend can't connect to backend
**Solution**:
- Verify backend CORS origins include your frontend URL
- Check browser console for exact error
- Ensure backend URL in frontend `.env.production` is correct

#### 2. Container App Won't Start
**Problem**: Container app shows "Provisioning failed"
**Solution**:
```bash
# Check logs
az containerapp logs show \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --tail 50
```

#### 3. Static Web App Build Fails
**Problem**: GitHub Action fails
**Solution**:
- Check GitHub Actions tab for error details
- Verify `app_location` and `output_location` in workflow
- Ensure `package.json` has correct build script

#### 4. API Returns 404
**Problem**: API endpoints return 404
**Solution**:
- Verify backend is running: `curl https://YOUR_BACKEND_URL/`
- Check API paths are correct (`/api/todos`)
- Review backend logs

#### 5. Environment Variables Not Working
**Problem**: `VITE_API_URL` is undefined
**Solution**:
- Ensure variable starts with `VITE_`
- Rebuild frontend after changing `.env` files
- For production, verify GitHub secrets are set

### Getting Help

1. **Backend Logs**:
   ```bash
   az containerapp logs show \
     --name $CONTAINER_APP_NAME \
     --resource-group $RESOURCE_GROUP \
     --tail 100 \
     --follow
   ```

2. **Frontend Logs**:
   - Check GitHub Actions tab in your repository
   - View deployment logs in Azure Portal → Static Web Apps

3. **Test Locally First**:
   - Always test changes locally before deploying
   - Use browser DevTools Network tab

---

## Next Steps & Enhancements

### Add Database (PostgreSQL)
- Azure Database for PostgreSQL has a free tier (Burstable B1MS)
- Replace in-memory storage with SQLAlchemy + PostgreSQL

### Add Authentication
- Integrate Azure AD B2C
- Add user-specific todos

### Add More Features
- Todo categories
- Due dates and reminders
- Search and filter
- Dark mode

### CI/CD Improvements
- Add automated tests
- Separate staging/production environments
- Database migrations

### Monitoring
- Application Insights for monitoring
- Custom dashboards in Azure Portal

---

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Azure Static Web Apps Docs](https://docs.microsoft.com/en-us/azure/static-web-apps/)
- [Azure Container Apps Docs](https://docs.microsoft.com/en-us/azure/container-apps/)
- [Vite Documentation](https://vitejs.dev/)

---

## Conclusion

Congratulations! You've built and deployed a full-stack application to Azure using modern technologies. This foundation can be extended to build production-ready applications.

**What you've learned**:
- Building REST APIs with FastAPI
- Creating React frontends with TypeScript
- Containerizing Python applications
- Deploying to Azure cloud platform
- Managing cloud resources

**Remember to**:
- Monitor your Azure costs
- Delete resources when not needed
- Keep dependencies updated
- Follow security best practices

Happy coding! 🚀
