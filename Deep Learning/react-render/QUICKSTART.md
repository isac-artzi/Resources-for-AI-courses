# ⚡ Quick Start Guide

Get the app running locally in 5 minutes!

---

## Prerequisites

- Python 3.8+
- Node.js 14+
- A terminal/command prompt

---

## 🚀 Start in 5 Minutes

### Terminal 1: Backend

```bash
# 1. Navigate to backend
cd backend

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows

# 4. Install dependencies (takes 2-3 minutes)
pip install -r requirements.txt

# 5. Start backend server
uvicorn app:app --reload --port 8000
```

✅ Backend running at: http://localhost:8000

---

### Terminal 2: Frontend

Open a NEW terminal:

```bash
# 1. Navigate to frontend
cd frontend

# 2. Install dependencies (takes 1-2 minutes)
npm install

# 3. Start frontend server
npm start
```

✅ Frontend running at: http://localhost:3000

Browser should open automatically!

---

## 🎯 Test It Out

1. **Open** http://localhost:3000
2. **Click** the upload area
3. **Select** any image from your computer
4. **Click** "Classify Image"
5. **See** AI predictions! 🎉

---

## 📦 What Gets Installed?

### Backend (~800MB)
- FastAPI: Web framework
- PyTorch: Deep learning
- Pillow: Image processing

### Frontend (~200MB)
- React: UI library
- Dependencies

**First install takes longer due to downloads**

---

## ⚠️ Common Issues

### "Port 8000 already in use"
```bash
# Use different port
uvicorn app:app --reload --port 8001

# Update frontend/src/services/api.js:
const API_URL = 'http://localhost:8001';
```

### "Module not found"
```bash
# Backend:
pip install -r requirements.txt

# Frontend:
npm install
```

### "Cannot connect to backend"
- Make sure backend is running (Terminal 1)
- Check it's on port 8000
- Visit http://localhost:8000 to verify

---

## 📖 Next Steps

1. ✅ Got it running? Read [TESTING_GUIDE.md](TESTING_GUIDE.md)
2. 🚀 Ready to deploy? Read [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
3. 📚 Want to learn more? Read [README.md](README.md)

---

## 🆘 Need Help?

1. Check the error message carefully
2. Search the error on Google
3. Ask instructor or classmates
4. Review the detailed guides

---

**That's it! You're running a full-stack AI app!** 🎊
