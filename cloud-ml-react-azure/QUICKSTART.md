# Quick Start Guide

Get the app running locally in under 5 minutes!

## 1. Prerequisites Check

```bash
# Check Node.js (need 18+)
node --version

# Check Python (need 3.9+)
python3 --version

# Check Git
git --version
```

If any are missing, install them from the links in README.md.

## 2. Backend Setup (Terminal 1)

```bash
# Navigate to backend
cd backend

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Test it**: Open `http://localhost:8000/docs` in your browser

## 3. Frontend Setup (Terminal 2 - New Terminal)

```bash
# Navigate to frontend (from project root)
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

You should see:
```
  VITE v5.0.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
```

**Test it**: Open `http://localhost:5173` in your browser

## 4. Try It Out!

1. Open `http://localhost:5173`
2. You should see the Todo app
3. Try creating a todo
4. Mark it as complete
5. Delete it

## Common Issues

### Backend won't start
```bash
# Make sure you're in the backend directory
pwd  # Should show .../backend

# Make sure virtual environment is activated
# You should see (venv) in your terminal prompt
```

### Frontend won't start
```bash
# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### Can't create todos (CORS error)
- Check that backend is running on port 8000
- Check browser console for exact error
- Verify frontend/.env.local has: `VITE_API_URL=http://localhost:8000`

## Next Steps

- Read [TUTORIAL.md](./TUTORIAL.md) for detailed explanations
- Explore the code in `backend/app/` and `frontend/src/`
- Try modifying the UI or adding features
- Deploy to Azure following the tutorial

## Stopping the Servers

**Backend**: Press `Ctrl+C` in Terminal 1
**Frontend**: Press `Ctrl+C` in Terminal 2

To deactivate Python virtual environment:
```bash
deactivate
```

---

Need help? Check [TUTORIAL.md](./TUTORIAL.md) for detailed troubleshooting!
