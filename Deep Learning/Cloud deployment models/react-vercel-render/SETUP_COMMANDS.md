# Setup Commands Reference

Quick reference for all commands needed to set up and run the project.

## Table of Contents
- [Initial Setup](#initial-setup)
- [Backend Commands](#backend-commands)
- [Frontend Commands](#frontend-commands)
- [Testing Commands](#testing-commands)
- [Deployment Commands](#deployment-commands)
- [Troubleshooting Commands](#troubleshooting-commands)

---

## Initial Setup

### Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### System Requirements Check
```bash
# Check Python version (need 3.8+)
python --version

# Check Node.js version (need 16+)
node --version

# Check npm version
npm --version

# Check pip
pip --version
```

---

## Backend Commands

### Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment (macOS/Linux)
source venv/bin/activate

# Activate virtual environment (Windows)
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install specific TensorFlow version (if needed)
pip install tensorflow-cpu==2.15.0  # CPU-only (faster)

# Create .env file
cp .env.example .env
```

### Running

```bash
# Start development server (auto-reload)
python main.py

# Alternative: using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Start on different port
uvicorn main:app --reload --port 8001

# Production mode (no reload)
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Testing

```bash
# Run API tests
python test_api.py

# Test specific endpoint with curl
curl http://localhost:8000/health
curl http://localhost:8000/
curl http://localhost:8000/model/info

# Test prediction endpoint
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

### Package Management

```bash
# List installed packages
pip list

# Freeze dependencies
pip freeze > requirements.txt

# Update all packages
pip install --upgrade -r requirements.txt

# Install new package
pip install package-name

# Uninstall package
pip uninstall package-name
```

### Cleanup

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv

# Clear Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

---

## Frontend Commands

### Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Or with yarn
yarn install

# Create .env file
cp .env.example .env
```

### Running

```bash
# Start development server (port 3000)
npm start

# Start on different port
PORT=3001 npm start

# Start with custom host
HOST=0.0.0.0 npm start
```

### Building

```bash
# Create production build
npm run build

# Test production build locally
npx serve -s build

# Build with custom env
REACT_APP_API_URL=https://api.example.com npm run build
```

### Testing

```bash
# Run tests
npm test

# Run tests with coverage
npm test -- --coverage

# Run tests in watch mode
npm test -- --watch
```

### Package Management

```bash
# List installed packages
npm list --depth=0

# Update packages
npm update

# Install new package
npm install package-name

# Install dev dependency
npm install --save-dev package-name

# Uninstall package
npm uninstall package-name

# Check for outdated packages
npm outdated

# Audit for vulnerabilities
npm audit

# Fix vulnerabilities
npm audit fix
```

### Cleanup

```bash
# Remove node_modules
rm -rf node_modules

# Remove build
rm -rf build

# Clear npm cache
npm cache clean --force

# Reinstall everything
rm -rf node_modules package-lock.json
npm install
```

---

## Testing Commands

### Full Application Test

```bash
# Terminal 1: Start backend
cd backend
source venv/bin/activate
python main.py

# Terminal 2: Start frontend
cd frontend
npm start

# Terminal 3: Run backend tests
cd backend
python test_api.py
```

### API Testing with cURL

```bash
# Health check
curl http://localhost:8000/health

# Get model info
curl http://localhost:8000/model/info

# Upload image
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/image.jpg"

# Pretty print JSON
curl -s http://localhost:8000/health | python -m json.tool
```

### API Testing with HTTPie (alternative)

```bash
# Install httpie
pip install httpie

# Test endpoints
http GET http://localhost:8000/health
http POST http://localhost:8000/predict file@image.jpg
```

### Load Testing

```bash
# Install Apache Bench (macOS)
brew install httpd

# Simple load test
ab -n 100 -c 10 http://localhost:8000/health

# Detailed load test
ab -n 1000 -c 50 -g results.tsv http://localhost:8000/
```

---

## Deployment Commands

### Git Commands

```bash
# Initialize repository
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit"

# Add remote
git remote add origin https://github.com/USER/REPO.git

# Push to GitHub
git push -u origin main

# Check status
git status

# View changes
git diff

# Create branch
git checkout -b feature-name

# Merge branch
git checkout main
git merge feature-name
```

### Vercel Deployment

```bash
# Install Vercel CLI
npm install -g vercel

# Login to Vercel
vercel login

# Deploy (from frontend directory)
cd frontend
vercel

# Deploy to production
vercel --prod

# Check deployment status
vercel ls

# View logs
vercel logs

# Set environment variable
vercel env add REACT_APP_API_URL
```

### Render Deployment

```bash
# No CLI needed - deploy via dashboard
# Or use render.yaml configuration

# Check deployment via curl
curl https://your-app.onrender.com/health
```

---

## Troubleshooting Commands

### Find and Kill Processes

```bash
# Find process on port 8000 (macOS/Linux)
lsof -i :8000

# Find process on port 3000
lsof -i :3000

# Kill process by PID
kill -9 <PID>

# Find and kill (one command)
lsof -ti:8000 | xargs kill -9

# Windows: Find process on port
netstat -ano | findstr :8000

# Windows: Kill process
taskkill /PID <PID> /F
```

### System Information

```bash
# Check Python path
which python

# Check Node path
which node

# Check system resources (macOS/Linux)
top

# Check disk space
df -h

# Check memory
free -m  # Linux
vm_stat  # macOS
```

### Logs and Debugging

```bash
# Backend: View logs
python main.py 2>&1 | tee backend.log

# Frontend: View logs
npm start 2>&1 | tee frontend.log

# Backend: Verbose logging
uvicorn main:app --reload --log-level debug

# View last 50 lines of log
tail -n 50 backend.log

# Follow log in real-time
tail -f backend.log
```

### Network Testing

```bash
# Check if port is open
nc -zv localhost 8000

# Check API from external device
curl http://YOUR_IP:8000/health

# Get your IP address
# macOS/Linux:
ifconfig | grep "inet "
# Windows:
ipconfig
```

### Dependency Issues

```bash
# Backend: Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Frontend: Reinstall dependencies
rm -rf node_modules package-lock.json
npm install

# Backend: Install specific version
pip install tensorflow==2.15.0

# Frontend: Install specific version
npm install react@18.2.0
```

### Cache Clearing

```bash
# Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# npm cache
npm cache clean --force

# Browser cache
# Use Ctrl+Shift+R (hard refresh)
# Or clear in browser settings
```

---

## Complete Workflow Commands

### First Time Setup

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env

# Frontend (new terminal)
cd frontend
npm install
cp .env.example .env
```

### Daily Development

```bash
# Terminal 1: Backend
cd backend
source venv/bin/activate
python main.py

# Terminal 2: Frontend
cd frontend
npm start
```

### Before Committing

```bash
# Run tests
cd backend && python test_api.py
cd frontend && npm test

# Check for issues
npm run build  # Ensure build works

# Add and commit
git add .
git status
git commit -m "Description of changes"
git push
```

### Deployment

```bash
# 1. Commit and push to GitHub
git add .
git commit -m "Ready for deployment"
git push

# 2. Frontend deploys automatically on Vercel
# 3. Backend deploys automatically on Render
# 4. Verify deployments
curl https://your-backend.onrender.com/health
```

---

## Quick Reference

### Start Everything
```bash
# Backend
cd backend && source venv/bin/activate && python main.py

# Frontend (separate terminal)
cd frontend && npm start
```

### Stop Everything
```bash
# Press Ctrl+C in each terminal
```

### Reset Everything
```bash
# Backend
cd backend
deactivate
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Frontend
cd frontend
rm -rf node_modules package-lock.json
npm install
```

---

## Platform-Specific Commands

### macOS/Linux

```bash
# Activate venv
source venv/bin/activate

# Find process
lsof -i :8000

# System info
uname -a
```

### Windows

```bash
# Activate venv
venv\Scripts\activate

# Find process
netstat -ano | findstr :8000

# System info
systeminfo
```

---

## Environment Variables

### Backend (.env)
```bash
export PORT=8000
export ENVIRONMENT=development
```

### Frontend (.env)
```bash
export REACT_APP_API_URL=http://localhost:8000
```

### Load Environment Variables
```bash
# Load from .env file (if using python-dotenv)
# Automatic in Python with:
from dotenv import load_dotenv
load_dotenv()
```

---

## Help Commands

```bash
# Python help
python --help

# Pip help
pip --help

# Node help
node --help

# NPM help
npm --help

# Uvicorn help
uvicorn --help

# Git help
git --help
```

---

**Pro Tips:**

1. **Use aliases** for common commands:
   ```bash
   # Add to ~/.bashrc or ~/.zshrc
   alias be='cd backend && source venv/bin/activate'
   alias fe='cd frontend'
   alias runbe='cd backend && source venv/bin/activate && python main.py'
   alias runfe='cd frontend && npm start'
   ```

2. **Use tmux or screen** to manage multiple terminals

3. **Create a Makefile** for common tasks:
   ```makefile
   start-backend:
       cd backend && source venv/bin/activate && python main.py

   start-frontend:
       cd frontend && npm start
   ```

---

**Last Updated**: 2026
