# Testing Guide
## React + FastAPI Deep Learning Application

Complete guide for testing your application locally and in production.

## 📋 Table of Contents

1. [Local Testing](#local-testing)
2. [Backend Testing](#backend-testing)
3. [Frontend Testing](#frontend-testing)
4. [Integration Testing](#integration-testing)
5. [Production Testing](#production-testing)
6. [Common Test Cases](#common-test-cases)
7. [Troubleshooting Tests](#troubleshooting-tests)

## 🏠 Local Testing

### Prerequisites Check

Before starting, verify your environment:

```bash
# Check Node.js version (should be 18+)
node --version

# Check Python version (should be 3.8+)
python --version

# Check npm
npm --version

# Check git
git --version
```

### Initial Setup

1. **Clone Repository**
   ```bash
   git clone <your-repo-url>
   cd cloud-ml-react-vercel
   ```

2. **Setup Backend**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   # venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   cd ..
   ```

3. **Setup Frontend**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

## 🔧 Backend Testing

### Manual Testing

1. **Start Backend Server**
   ```bash
   cd backend
   source venv/bin/activate
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

   Expected output:
   ```
   INFO:     Uvicorn running on http://0.0.0.0:8000
   INFO:     Application startup complete.
   ```

2. **Test Health Endpoint**

   **Browser Method:**
   - Open: `http://localhost:8000/health`

   **curl Method:**
   ```bash
   curl http://localhost:8000/health
   ```

   Expected response:
   ```json
   {
     "status": "healthy",
     "model_loaded": true,
     "model_name": "MobileNetV2",
     "timestamp": "2024-01-15T10:30:00",
     "framework": "TensorFlow/Keras"
   }
   ```

3. **Test API Documentation**
   - Open: `http://localhost:8000/docs`
   - Should see interactive Swagger UI
   - Try the `/predict` endpoint directly

4. **Test Image Classification**

   Using curl:
   ```bash
   # Prepare a test image (download or use your own)
   curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/image.jpg"
   ```

   Using Python:
   ```python
   import requests

   url = "http://localhost:8000/predict"
   files = {"file": open("test_image.jpg", "rb")}
   response = requests.post(url, files=files)
   print(response.json())
   ```

### Automated Backend Tests

Create `backend/tests/test_api.py`:

```python
"""
Backend API Tests
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from PIL import Image
import io
import numpy as np

client = TestClient(app)

def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_predict_with_valid_image():
    """Test prediction with valid image"""
    # Create a test image
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    # Send request
    files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
    response = client.post("/predict", files=files)

    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "predictions" in data
    assert len(data["predictions"]) > 0

def test_predict_without_file():
    """Test prediction without file"""
    response = client.post("/predict")
    assert response.status_code == 422  # Unprocessable Entity

def test_predict_with_invalid_file():
    """Test prediction with non-image file"""
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
```

Run tests:
```bash
cd backend
pytest tests/ -v
```

### Model Testing

Create `backend/tests/test_model.py`:

```python
"""
Model Tests
"""
import pytest
import numpy as np
from app.model import ImageClassifier

@pytest.fixture
def classifier():
    """Create classifier instance"""
    return ImageClassifier()

def test_model_loads(classifier):
    """Test that model loads successfully"""
    assert classifier.model is not None
    assert classifier.model_name == "MobileNetV2"

def test_preprocess_image(classifier):
    """Test image preprocessing"""
    # Create random image
    img = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)

    # Preprocess
    processed = classifier.preprocess_image(img)

    # Verify output shape
    assert processed.shape == (1, 224, 224, 3)

def test_predict(classifier):
    """Test prediction"""
    # Create random image
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Get predictions
    predictions = classifier.predict(img)

    # Verify output
    assert len(predictions) > 0
    assert len(predictions) <= 5

    # Check prediction format
    for pred in predictions:
        assert len(pred) == 3  # (class_id, class_name, confidence)
        assert 0 <= pred[2] <= 1  # Confidence in [0, 1]
```

## ⚛️ Frontend Testing

### Manual Testing

1. **Start Frontend Server**
   ```bash
   cd frontend
   npm run dev
   ```

   Expected output:
   ```
   VITE v5.0.12  ready in XXX ms
   ➜  Local:   http://localhost:5173/
   ➜  press h to show help
   ```

2. **Test in Browser**
   - Open: `http://localhost:5173`
   - Should see the application interface
   - Check browser console for errors (F12)

3. **Test User Flow**

   **Step 1: Upload Image**
   - Click "Choose Image" or drag & drop
   - Select a JPG/PNG image
   - Verify preview appears

   **Step 2: Classify**
   - Click "Classify Image"
   - Should see loading spinner
   - Verify predictions appear

   **Step 3: Reset**
   - Click "Reset"
   - Verify everything clears

### Component Testing

Install testing dependencies:
```bash
cd frontend
npm install --save-dev @testing-library/react @testing-library/jest-dom vitest jsdom
```

Create `frontend/src/components/Results.test.jsx`:

```javascript
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import Results from './Results'

describe('Results Component', () => {
  it('renders predictions correctly', () => {
    const mockPredictions = [
      { class: 'Dog', confidence: 0.95, rank: 1 },
      { class: 'Cat', confidence: 0.03, rank: 2 }
    ]

    render(<Results predictions={mockPredictions} />)

    expect(screen.getByText('Dog')).toBeInTheDocument()
    expect(screen.getByText('Cat')).toBeInTheDocument()
  })

  it('shows no results message when empty', () => {
    render(<Results predictions={[]} />)
    expect(screen.getByText(/no predictions/i)).toBeInTheDocument()
  })
})
```

Run tests:
```bash
npm run test
```

### API Service Testing

Create `frontend/src/services/api.test.js`:

```javascript
import { describe, it, expect, vi } from 'vitest'
import axios from 'axios'
import { classifyImage } from './api'

// Mock axios
vi.mock('axios')

describe('API Service', () => {
  it('classifies image successfully', async () => {
    const mockResponse = {
      data: {
        success: true,
        predictions: [
          { class: 'Dog', confidence: 0.95 }
        ]
      }
    }

    axios.create.mockReturnValue({
      post: vi.fn().resolves(mockResponse)
    })

    const file = new File([''], 'test.jpg', { type: 'image/jpeg' })
    const result = await classifyImage(file)

    expect(result.success).toBe(true)
    expect(result.predictions).toHaveLength(1)
  })
})
```

## 🔗 Integration Testing

### Full Stack Test

1. **Start Both Servers**

   Terminal 1 (Backend):
   ```bash
   cd backend
   source venv/bin/activate
   uvicorn app.main:app --reload
   ```

   Terminal 2 (Frontend):
   ```bash
   cd frontend
   npm run dev
   ```

2. **Test Complete Flow**
   - Open `http://localhost:5173`
   - Upload an image
   - Click classify
   - Verify predictions appear

3. **Check Network Tab**
   - Open DevTools (F12)
   - Go to Network tab
   - Upload and classify image
   - Verify API call:
     - Request URL: `http://localhost:8000/predict`
     - Method: POST
     - Status: 200
     - Response includes predictions

### End-to-End Test Script

Create `test-e2e.sh`:

```bash
#!/bin/bash

echo "🧪 Running End-to-End Tests"
echo "======================================"

# Test Backend
echo "1. Testing Backend..."
HEALTH=$(curl -s http://localhost:8000/health)
if echo $HEALTH | grep -q "healthy"; then
  echo "✅ Backend is healthy"
else
  echo "❌ Backend health check failed"
  exit 1
fi

# Test Frontend
echo "2. Testing Frontend..."
FRONTEND=$(curl -s http://localhost:5173)
if [ -n "$FRONTEND" ]; then
  echo "✅ Frontend is running"
else
  echo "❌ Frontend is not accessible"
  exit 1
fi

# Test API Integration
echo "3. Testing API integration..."
# Add image classification test here

echo ""
echo "✅ All E2E tests passed!"
```

## 🌐 Production Testing

### After Deployment

1. **Test Backend Production**
   ```bash
   # Replace with your Railway URL
   curl https://your-app.railway.app/health
   ```

2. **Test Frontend Production**
   - Visit your Vercel URL
   - Test complete flow
   - Check browser console

3. **Test Cross-Origin Requests**
   - Ensure frontend can call backend
   - Check CORS headers in Network tab

### Performance Testing

1. **Backend Response Time**
   ```bash
   time curl -X POST "https://your-app.railway.app/predict" \
     -F "file=@test.jpg"
   ```

2. **Frontend Load Time**
   - Open DevTools > Network
   - Reload page
   - Check load time

### Load Testing

Using Apache Bench:
```bash
# Install ab (Apache Bench)
# Test with 100 requests, 10 concurrent
ab -n 100 -c 10 https://your-app.railway.app/health
```

Using curl-loader (Python):
```python
import concurrent.futures
import requests
import time

def test_prediction():
    files = {"file": open("test.jpg", "rb")}
    response = requests.post(
        "https://your-app.railway.app/predict",
        files=files
    )
    return response.status_code

# Run 10 concurrent requests
start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(test_prediction) for _ in range(10)]
    results = [f.result() for f in futures]

duration = time.time() - start
print(f"Completed 10 requests in {duration:.2f}s")
print(f"Success rate: {results.count(200)/len(results)*100}%")
```

## ✅ Common Test Cases

### Backend Test Cases

| Test Case | Expected Result |
|-----------|----------------|
| GET `/` | 200, returns API info |
| GET `/health` | 200, model_loaded: true |
| POST `/predict` with valid image | 200, returns predictions |
| POST `/predict` without file | 422, validation error |
| POST `/predict` with invalid file | 400, file type error |
| POST `/predict` with large file | 413 or timeout |

### Frontend Test Cases

| Test Case | Expected Result |
|-----------|----------------|
| Load homepage | Page renders without errors |
| Upload valid image | Preview shows image |
| Upload invalid file | Error message displayed |
| Classify without image | Button disabled |
| Classify valid image | Predictions displayed |
| Reset after classification | All state cleared |

### Integration Test Cases

| Test Case | Expected Result |
|-----------|----------------|
| Backend offline | Frontend shows connection error |
| Invalid API URL | Error message |
| Slow response | Loading spinner shows |
| Network error | User-friendly error message |

## 🐛 Troubleshooting Tests

### Backend Issues

**Issue**: Model not loading
```bash
# Check TensorFlow installation
python -c "import tensorflow as tf; print(tf.__version__)"

# Check model downloads
ls ~/.keras/models/
```

**Issue**: CORS errors
```python
# In backend/app/main.py, ensure:
allow_origins=["http://localhost:5173"]
```

### Frontend Issues

**Issue**: API calls fail
```javascript
// Check .env.local
console.log(import.meta.env.VITE_API_URL)
// Should print: http://localhost:8000
```

**Issue**: Build fails
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

## 📊 Test Coverage

### Check Backend Coverage

```bash
cd backend
pytest --cov=app tests/
```

### Check Frontend Coverage

```bash
cd frontend
npm run test -- --coverage
```

## 🎯 Testing Checklist

Before deploying:

- [ ] All backend endpoints respond correctly
- [ ] Model loads and predicts accurately
- [ ] Frontend builds without errors
- [ ] Image upload works
- [ ] Classification displays results
- [ ] Error handling works
- [ ] CORS configured correctly
- [ ] Environment variables set
- [ ] Production build tested locally
- [ ] Performance is acceptable

## 📚 Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [React Testing Library](https://testing-library.com/react)
- [Vitest Documentation](https://vitest.dev/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)

---

**Need Help?** Check logs, console errors, and refer to [README.md](README.md) for setup instructions.
