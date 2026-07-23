# 🏗️ Architecture Documentation

Understanding how all the pieces fit together.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         USER                                 │
│                      (Web Browser)                           │
└────────────────┬────────────────────────────────────────────┘
                 │
                 │ HTTPS
                 │
┌────────────────▼────────────────────────────────────────────┐
│                   FRONTEND (React)                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Components:                                          │   │
│  │  - ImageUpload: File selection & preview             │   │
│  │  - ResultDisplay: Show predictions                   │   │
│  │  - App: Main coordinator                             │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Services:                                            │   │
│  │  - api.js: Communicate with backend                  │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────┬────────────────────────────────────────────┘
                 │
                 │ HTTP POST /predict
                 │ (FormData with image)
                 │
┌────────────────▼────────────────────────────────────────────┐
│                   BACKEND (FastAPI)                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Endpoints:                                           │   │
│  │  - GET  /         : Welcome message                  │   │
│  │  - GET  /health   : Health check                     │   │
│  │  - POST /predict  : Image classification             │   │
│  │  - GET  /docs     : API documentation                │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Model (model.py):                                    │   │
│  │  - Load ResNet-18                                     │   │
│  │  - Preprocess image                                   │   │
│  │  - Make predictions                                   │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────┬────────────────────────────────────────────┘
                 │
                 │
┌────────────────▼────────────────────────────────────────────┐
│              DEEP LEARNING MODEL                             │
│                                                               │
│  ResNet-18 (PyTorch)                                         │
│  - Pre-trained on ImageNet                                   │
│  - 1000 categories                                           │
│  - ~45 MB model size                                         │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### 1. Image Upload Flow

```
User selects image
       ↓
ImageUpload component
       ↓
Creates preview (URL.createObjectURL)
       ↓
User clicks "Classify"
       ↓
App.handleImageUpload()
       ↓
api.uploadImage(file)
       ↓
Creates FormData
       ↓
POST /predict with image
       ↓
Backend receives image
       ↓
Validates file (type, size)
       ↓
Opens image with PIL
       ↓
Sends to model.predict()
       ↓
Model preprocesses (resize, normalize)
       ↓
Neural network forward pass
       ↓
Get top 5 predictions
       ↓
Return JSON response
       ↓
Frontend receives predictions
       ↓
ResultDisplay shows results
```

### 2. Detailed Request/Response

**Request (Frontend → Backend):**
```http
POST /predict HTTP/1.1
Host: localhost:8000
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary...

------WebKitFormBoundary...
Content-Disposition: form-data; name="file"; filename="cat.jpg"
Content-Type: image/jpeg

[Binary image data]
------WebKitFormBoundary...--
```

**Response (Backend → Frontend):**
```json
{
  "success": true,
  "filename": "cat.jpg",
  "predictions": [
    {
      "label": "tabby cat",
      "confidence": 0.9234,
      "class_id": 281
    },
    {
      "label": "Egyptian cat",
      "confidence": 0.0543,
      "class_id": 285
    },
    ...
  ],
  "message": "Image classified successfully"
}
```

---

## Component Breakdown

### Frontend Components

#### 1. App.js (Root Component)
**Purpose**: Main application coordinator

**State:**
- `loading`: Boolean (upload in progress)
- `result`: Object (prediction results)
- `error`: String (error messages)
- `backendStatus`: String (backend connection status)

**Key Functions:**
- `handleImageUpload()`: Process image upload
- `useEffect()`: Check backend health on mount

**Renders:**
- Header with title and status
- ImageUpload component
- ResultDisplay component
- Footer

---

#### 2. ImageUpload.js
**Purpose**: Handle file selection and upload

**State:**
- `selectedFile`: File object
- `previewUrl`: String (image preview URL)
- `dragActive`: Boolean (drag-and-drop state)

**Key Functions:**
- `handleFileChange()`: Process file selection
- `handleDragDrop()`: Handle drag-and-drop
- `processFile()`: Validate and create preview
- `handleUpload()`: Trigger upload via prop

**Features:**
- Drag-and-drop support
- Image preview
- File validation
- Upload button

---

#### 3. ResultDisplay.js
**Purpose**: Display classification results

**Props:**
- `result`: Prediction results object
- `error`: Error message string

**Features:**
- Conditional rendering (success/error)
- Top predictions list
- Confidence bars
- Color-coded confidence levels

---

#### 4. api.js (Service)
**Purpose**: Backend communication

**Functions:**
- `checkHealth()`: GET /health
- `uploadImage()`: POST /predict
- `getModelInfo()`: GET /model-info

**Configuration:**
- API URL (development/production)
- Error handling
- Request/response processing

---

### Backend Components

#### 1. app.py (Main Application)
**Purpose**: FastAPI application and routes

**Endpoints:**
```python
GET  /           # Welcome message
GET  /health     # Health check
POST /predict    # Image classification
GET  /model-info # Model information
```

**Key Features:**
- CORS middleware (cross-origin requests)
- File upload handling
- Error handling
- Logging

---

#### 2. model.py (ML Model)
**Purpose**: Deep learning model wrapper

**Class: ImageClassifier**

**Methods:**
- `__init__()`: Load model and setup
- `predict()`: Classify image
- `predict_from_path()`: Classify from file path

**Model Details:**
- Architecture: ResNet-18
- Framework: PyTorch
- Input: 224x224 RGB images
- Output: 1000 class probabilities
- Pre-trained: ImageNet weights

**Preprocessing:**
1. Resize to 256px
2. Center crop to 224x224
3. Convert to tensor
4. Normalize (ImageNet stats)

---

## Technology Stack

### Frontend

| Technology | Purpose | Version |
|------------|---------|---------|
| React | UI Framework | 18.2.0 |
| JavaScript | Programming Language | ES6+ |
| CSS3 | Styling | - |
| Fetch API | HTTP Requests | Native |

### Backend

| Technology | Purpose | Version |
|------------|---------|---------|
| Python | Programming Language | 3.8+ |
| FastAPI | Web Framework | 0.104.1 |
| Uvicorn | ASGI Server | 0.24.0 |
| PyTorch | Deep Learning | 2.1.0 |
| Pillow | Image Processing | 10.1.0 |

### Deployment

| Service | Purpose | Plan |
|---------|---------|------|
| Render | Backend Hosting | Free |
| Render | Frontend Hosting | Free |
| GitHub | Version Control | Free |

---

## File Structure

```
react-render/
│
├── README.md                    # Main documentation
├── QUICKSTART.md               # Quick start guide
├── TESTING_GUIDE.md            # Testing instructions
├── DEPLOYMENT_GUIDE.md         # Deployment instructions
├── ARCHITECTURE.md             # This file
├── .gitignore                  # Git ignore rules
│
├── backend/                    # FastAPI backend
│   ├── app.py                 # Main application
│   ├── model.py               # ML model wrapper
│   ├── requirements.txt       # Python dependencies
│   ├── render.yaml           # Render config
│   └── imagenet_classes.txt  # Class labels
│
└── frontend/                  # React frontend
    ├── public/
    │   └── index.html        # HTML template
    ├── src/
    │   ├── index.js          # Entry point
    │   ├── index.css         # Global styles
    │   ├── App.js            # Main component
    │   ├── App.css           # App styles
    │   ├── reportWebVitals.js # Performance
    │   ├── components/
    │   │   ├── ImageUpload.js
    │   │   ├── ImageUpload.css
    │   │   ├── ResultDisplay.js
    │   │   └── ResultDisplay.css
    │   └── services/
    │       └── api.js        # API service
    ├── package.json          # Node dependencies
    └── .env.example         # Environment template
```

---

## Security Considerations

### Frontend

✅ **Good Practices:**
- No sensitive data in code
- Environment variables for configuration
- Input validation
- HTTPS in production

⚠️ **Limitations:**
- All frontend code is public
- Don't store secrets in frontend
- API keys visible to users

### Backend

✅ **Good Practices:**
- CORS configuration
- File type validation
- File size limits
- Error handling
- Input sanitization

⚠️ **Considerations:**
- No authentication (tutorial simplicity)
- No rate limiting
- No file storage (temporary processing)

### Production Recommendations

For a production app, add:
1. User authentication (JWT)
2. Rate limiting
3. File upload limits
4. Request validation
5. Logging and monitoring
6. Database for results
7. CDN for static files
8. Caching layer

---

## Performance Characteristics

### Frontend

**Initial Load:**
- HTML: < 1KB
- JavaScript bundle: ~300KB (gzipped)
- CSS: ~10KB
- Total: ~300KB
- Load time: 1-3 seconds (depends on network)

**Image Upload:**
- File size: Typically 500KB - 5MB
- Upload time: 1-5 seconds (depends on network)

### Backend

**Cold Start (Free Tier):**
- Service spin-up: 10-15 seconds
- Model loading: 5-10 seconds
- Total: 20-30 seconds
- Only after 15 min inactivity

**Warm Response:**
- Image preprocessing: 100-200ms
- Model inference: 50-150ms
- Total: 200-500ms
- Very fast for subsequent requests

**Memory Usage:**
- Base application: ~100MB
- PyTorch model: ~300MB
- Per request: ~50MB
- Total: ~450MB (fits in 512MB free tier)

---

## Scalability

### Current Limitations (Free Tier)

- **Backend**:
  - 512MB RAM
  - Single instance
  - Sleeps after 15 min
  - ~100 requests/day practical limit

- **Frontend**:
  - Unlimited traffic
  - Global CDN
  - No sleep

### Scaling Strategies

**For More Users:**
1. Upgrade to paid plan ($7/month)
   - More RAM
   - No sleep
   - Better CPU

2. Add caching
   - Redis for results
   - Reduce model calls

3. Optimize model
   - Use quantization
   - Smaller model (MobileNet)

4. Add load balancer
   - Multiple instances
   - Distribute traffic

5. Use cloud ML services
   - AWS SageMaker
   - Google Cloud AI
   - Azure ML

---

## Extension Ideas

### Easy Enhancements

1. **UI Improvements**
   - Dark mode toggle
   - Better animations
   - Mobile optimization

2. **Features**
   - Save classification history
   - Share results
   - Compare multiple images

3. **Analytics**
   - Track popular images
   - Usage statistics
   - Error monitoring

### Advanced Enhancements

1. **Custom Model**
   - Train on specific dataset
   - Fine-tune for domain
   - Support multiple models

2. **User Accounts**
   - Authentication
   - Personal history
   - Saved favorites

3. **Batch Processing**
   - Upload multiple images
   - Process in background
   - Email results

4. **API Features**
   - API keys
   - Rate limiting
   - Usage quotas
   - Webhooks

---

## Learning Resources

### React
- [Official Tutorial](https://react.dev/learn)
- [React Hooks](https://react.dev/reference/react)
- [Component Patterns](https://react.dev/learn/thinking-in-react)

### FastAPI
- [Official Docs](https://fastapi.tiangolo.com)
- [Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Advanced Topics](https://fastapi.tiangolo.com/advanced/)

### PyTorch
- [Tutorials](https://pytorch.org/tutorials/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Transfer Learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

### Cloud Deployment
- [Render Docs](https://render.com/docs)
- [12-Factor App](https://12factor.net/)
- [Cloud Architecture](https://cloud.google.com/architecture)

---

## Conclusion

This architecture demonstrates:

✅ Modern web development (React + FastAPI)
✅ Deep learning integration (PyTorch)
✅ Cloud deployment (Render)
✅ RESTful API design
✅ Component-based frontend
✅ Separation of concerns

**Great foundation for building more complex applications!** 🚀
