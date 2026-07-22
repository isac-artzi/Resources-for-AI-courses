# Quick Start Guide

Get the app running in 3 minutes!

## Step 1: Install Dependencies

```bash
cd cloud-ml-react-local
npm install
```

Wait for installation to complete (~2 minutes).

## Step 2: Start the App

```bash
npm start
```

Your browser will open automatically to `http://localhost:3000`.

## Step 3: Use the App

1. **Train the Model**
   - Click the "Train Model" button
   - Wait 30-60 seconds
   - Watch accuracy improve each epoch

2. **Draw a Digit**
   - Draw any digit (0-9) on the canvas
   - Use your mouse or finger

3. **See Predictions**
   - The model predicts what you drew
   - View confidence scores and probabilities

4. **Try Again**
   - Click "Clear Canvas"
   - Draw a different digit

## That's It!

For detailed explanations, see:
- **README.md** - Project overview and features
- **TUTORIAL.md** - Complete tutorial with exercises

## Troubleshooting

### "npm: command not found"
Install Node.js from [nodejs.org](https://nodejs.org/)

### "Port 3000 already in use"
```bash
PORT=3001 npm start
```

### Model training is slow
Reduce training data in `src/utils/modelUtils.js`:
```javascript
const trainDataSize = 1000; // Line 123
```

### Need Help?
Check console for errors: Press F12 → Console tab

---

**Happy Learning! 🚀**
