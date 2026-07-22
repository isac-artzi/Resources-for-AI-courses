# Quick Start Guide

Get your React/FastAPI deep learning app running in 10 minutes.

## Prerequisites

- Python 3.8+
- Node.js 16+
- Git

## Setup (5 minutes)

### 1. Clone/Download Project

```bash
cd /path/to/project
```

### 2. Setup Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Setup Frontend

```bash
cd ../frontend
npm install
cp .env.example .env
```

## Run Locally (2 minutes)

### Terminal 1 - Backend:

```bash
cd backend
source venv/bin/activate
python main.py
```

Wait for: `Model loaded successfully`

### Terminal 2 - Frontend:

```bash
cd frontend
npm start
```

Browser opens automatically to `http://localhost:3000`

## Test It (3 minutes)

1. Check "API Connected" status appears
2. Click "Choose Image" and select any image
3. Click "Classify Image"
4. View predictions!

## Common Issues

**Port 8000 in use?**
```bash
# Kill process
lsof -i :8000  # macOS/Linux
kill -9 <PID>
```

**TensorFlow installation slow?**
```bash
# Use CPU-only version
pip install tensorflow-cpu==2.15.0
```

**CORS error?**
- Verify backend includes `http://localhost:3000` in CORS origins
- Restart backend

## Next Steps

- See [TESTING_GUIDE.md](TESTING_GUIDE.md) for detailed testing
- See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for cloud deployment
- See [README.md](README.md) for complete documentation

## Deploy to Cloud

**Backend (Render):**
1. Push to GitHub
2. Go to render.com
3. Create Web Service
4. Connect GitHub repo
5. Set root directory: `backend`

**Frontend (Vercel):**
1. Go to vercel.com
2. Import GitHub repo
3. Set root directory: `frontend`
4. Add env var: `REACT_APP_API_URL=<your-render-url>`

Done! 🚀

## Help

Issues? Check:
- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Troubleshooting
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Deployment help
- Backend logs for errors
- Browser console (F12) for frontend errors
