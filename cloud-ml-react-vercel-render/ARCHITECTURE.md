# Architecture Documentation

Detailed architecture overview of the React/FastAPI Deep Learning application.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Cloud Layer                              │
│  ┌──────────────────┐              ┌──────────────────┐         │
│  │  Vercel (CDN)    │              │  Render.com      │         │
│  │  - React App     │◄───HTTPS────►│  - FastAPI       │         │
│  │  - Static Files  │              │  - Python 3.9    │         │
│  │  - Auto Deploy   │              │  - Uvicorn       │         │
│  └──────────────────┘              └──────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
         │                                      │
         │ HTTPS                                │ REST API
         ▼                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Application Layer                           │
│  ┌──────────────────┐              ┌──────────────────┐         │
│  │   React SPA      │              │   FastAPI App    │         │
│  │  - Components    │              │  - Endpoints     │         │
│  │  - State Mgmt    │              │  - CORS          │         │
│  │  - Axios Client  │              │  - Validation    │         │
│  └──────────────────┘              └──────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                                   ┌──────────────────┐
                                   │  Model Layer     │
                                   │  - TensorFlow    │
                                   │  - MobileNetV2   │
                                   │  - Preprocessing │
                                   └──────────────────┘
```

## Technology Stack

### Frontend
- **Framework**: React 18
- **Build Tool**: Create React App
- **HTTP Client**: Axios
- **Styling**: CSS3 (no framework)
- **Hosting**: Vercel

### Backend
- **Framework**: FastAPI 0.109
- **Server**: Uvicorn
- **Deep Learning**: TensorFlow 2.15
- **Image Processing**: Pillow, NumPy
- **Hosting**: Render.com

## Component Architecture

### Frontend Components

```
App.js (Root Component)
├── ImageUpload.js (Upload & Preview)
│   ├── File Input
│   ├── Image Preview
│   └── Upload Button
└── Results.js (Display Predictions)
    ├── Prediction List
    ├── Confidence Bars
    └── Top Prediction Highlight
```

### Backend Modules

```
main.py (FastAPI Application)
├── Endpoints
│   ├── GET /
│   ├── GET /health
│   ├── POST /predict
│   ├── POST /predict/batch
│   └── GET /model/info
├── CORS Middleware
└── Startup Events

model.py (Model Handler)
├── ModelHandler Class
│   ├── load_model()
│   ├── preprocess_image()
│   ├── predict()
│   └── get_model_info()
└── Alternative Handlers
    ├── CustomModelHandler
    └── PyTorchModelHandler
```

## Data Flow

### Image Classification Flow

```
1. User selects image
   └─► Frontend validates file (type, size)
       └─► Creates preview

2. User clicks "Classify"
   └─► FormData created with image
       └─► Axios POST to /predict

3. Backend receives request
   └─► FastAPI validates file
       └─► Converts to PIL Image
           └─► Preprocesses image
               └─► Model inference
                   └─► Decodes predictions

4. Response sent to frontend
   └─► JSON with predictions
       └─► Frontend displays results
```

### Request/Response Format

**Request:**
```http
POST /predict HTTP/1.1
Host: backend-url.onrender.com
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary

------WebKitFormBoundary
Content-Disposition: form-data; name="file"; filename="dog.jpg"
Content-Type: image/jpeg

[binary image data]
------WebKitFormBoundary--
```

**Response:**
```json
{
  "success": true,
  "filename": "dog.jpg",
  "predictions": [
    {
      "class": "golden_retriever",
      "confidence": 0.95,
      "class_id": "n02099601"
    },
    {
      "class": "labrador",
      "confidence": 0.03,
      "class_id": "n02099712"
    }
  ]
}
```

## Security Architecture

### CORS Configuration

```python
allowed_origins = [
    "http://localhost:3000",      # Development
    "https://*.vercel.app",       # Production
]
```

### File Validation

**Frontend:**
- File type check (MIME type)
- File size limit (10MB)
- Client-side preview generation

**Backend:**
- Content-Type validation
- PIL Image verification
- File size limits
- Error handling

### Environment Variables

**Frontend (.env):**
```
REACT_APP_API_URL=https://api.example.com
```

**Backend (.env):**
```
PORT=8000
ENVIRONMENT=production
ALLOWED_ORIGINS=https://app.example.com
```

## Deep Learning Model

### MobileNetV2 Specifications

- **Architecture**: MobileNetV2
- **Weights**: ImageNet (pre-trained)
- **Input Size**: 224x224x3
- **Output**: 1000 classes
- **Framework**: TensorFlow/Keras

### Preprocessing Pipeline

```python
1. Resize to 224x224
2. Convert to array
3. Add batch dimension [1, 224, 224, 3]
4. Normalize (MobileNetV2 preprocessing)
5. Feed to model
```

### Inference Pipeline

```python
1. Model.predict() → Raw logits
2. decode_predictions() → Class names
3. Sort by confidence
4. Return top 5 predictions
```

## Deployment Architecture

### Vercel Deployment

```
GitHub Push
    ▼
Vercel Webhook
    ▼
Build Process
├── npm install
├── npm run build
└── Create static bundle
    ▼
Deploy to CDN
├── Edge Network
└── Automatic HTTPS
```

### Render Deployment

```
GitHub Push
    ▼
Render Webhook
    ▼
Build Process
├── pip install -r requirements.txt
└── Download TensorFlow & model weights
    ▼
Start Server
├── uvicorn main:app --port $PORT
└── Health checks
    ▼
Live on Render
```

## Performance Considerations

### Frontend Optimizations

- **Code Splitting**: React lazy loading (can be added)
- **Image Optimization**: Client-side compression
- **Caching**: Browser caching for static assets
- **CDN**: Vercel edge network

### Backend Optimizations

- **Model Loading**: Load once on startup
- **Connection Pooling**: Uvicorn workers (production)
- **Response Compression**: gzip middleware
- **Caching**: Model predictions cache (optional)

### Latency Breakdown

Typical request latency:
```
Frontend → Backend:     50-200ms  (network)
Backend Processing:     100-500ms (inference)
Backend → Frontend:     50-200ms  (network)
────────────────────────────────────────────
Total:                  200-900ms
```

Free tier considerations:
- Render cold start: +10-30s (first request)
- Vercel is always fast (CDN)

## Scalability

### Horizontal Scaling

**Frontend:**
- Automatic with Vercel CDN
- No configuration needed

**Backend:**
- Add Render instances
- Load balancer (automatic)
- Shared-nothing architecture

### Vertical Scaling

**Backend Options:**
- Free: 512MB RAM, 0.5 CPU
- Starter: 512MB RAM, 0.5 CPU (always-on)
- Standard: 2GB RAM, 1 CPU
- Pro: 4GB RAM, 2 CPU

### Database Integration (Future)

```
FastAPI Backend
    ├── PostgreSQL (Render)
    │   ├── User data
    │   ├── Prediction history
    │   └── Analytics
    └── Redis (optional)
        └── Prediction cache
```

## Monitoring & Logging

### Logs

**Frontend:**
- Browser console logs
- Vercel runtime logs
- Error boundaries

**Backend:**
- Python logging module
- Uvicorn access logs
- Render dashboard logs

### Metrics to Monitor

- **Frontend**: Page load time, API response time
- **Backend**: Request rate, error rate, model latency
- **Infrastructure**: CPU, memory, disk usage

## Error Handling

### Frontend Error Handling

```javascript
try {
  // API request
} catch (err) {
  if (err.response) {
    // Server error (4xx, 5xx)
  } else if (err.request) {
    // Network error
  } else {
    // Client error
  }
}
```

### Backend Error Handling

```python
try:
  # Process image
except HTTPException:
  # Re-raise HTTP exceptions
except Exception as e:
  # Log and return 500
  raise HTTPException(500, detail=str(e))
```

## Future Enhancements

### Planned Features

1. **Authentication**:
   - User accounts
   - JWT tokens
   - Protected endpoints

2. **Database**:
   - Store predictions
   - User history
   - Analytics

3. **Batch Processing**:
   - Multiple image upload
   - Background processing
   - Queue system (Celery/Redis)

4. **Custom Models**:
   - Upload custom models
   - Model versioning
   - A/B testing

5. **Advanced Features**:
   - Real-time inference
   - Video processing
   - Model fine-tuning API

## Development Workflow

```
1. Local Development
   ├── Backend: localhost:8000
   └── Frontend: localhost:3000

2. Git Commit & Push
   └── Push to GitHub

3. Automatic Deployment
   ├── Vercel: Frontend deploy
   └── Render: Backend deploy

4. Testing
   ├── Health checks
   └── Integration tests

5. Monitoring
   └── Check logs & metrics
```

## Security Best Practices

1. **API Security**:
   - CORS configuration
   - Rate limiting (add middleware)
   - Input validation
   - File size limits

2. **Secret Management**:
   - Environment variables
   - No secrets in code
   - Vercel/Render secret management

3. **HTTPS**:
   - Automatic with Vercel
   - Automatic with Render
   - No HTTP in production

## License & Attribution

- TensorFlow: Apache 2.0
- FastAPI: MIT
- React: MIT
- MobileNetV2: Apache 2.0

---

**Version**: 1.0.0
**Last Updated**: 2026
**Maintainer**: Your Team
