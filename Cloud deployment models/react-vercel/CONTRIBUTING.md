# Contributing Guide
## For Contributors

This guide helps you understand, modify, and extend the project.

## 📚 Understanding the Codebase

### Project Structure

```
react-vercel/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── main.py            # 🔥 START HERE: Main API routes
│   │   ├── model.py           # ML model wrapper
│   │   └── utils.py           # Helper functions
│   └── requirements.txt       # Python dependencies
│
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── App.jsx            # 🔥 START HERE: Main component
│   │   ├── components/        # Reusable components
│   │   └── services/api.js    # API integration
│   └── package.json           # Node dependencies
│
└── Documentation files
```

### Reading Order for Students

1. **Start with README.md** - Understand the big picture
2. **Backend**:
   - `backend/app/main.py` - See how API endpoints work
   - `backend/app/model.py` - Understand ML model integration
   - `backend/app/utils.py` - Learn helper patterns
3. **Frontend**:
   - `frontend/src/App.jsx` - See React component structure
   - `frontend/src/services/api.js` - Learn API communication
   - `frontend/src/components/` - Study reusable components

## 🎓 Learning Exercises

### Beginner Level

1. **Change UI Colors**
   - File: `frontend/src/App.css`
   - Modify CSS variables in `:root`
   - See changes in real-time

2. **Add New Prediction Fields**
   - File: `frontend/src/components/Results.jsx`
   - Display additional info (processing time, model name)

3. **Customize Error Messages**
   - File: `backend/app/main.py`
   - Make error messages more user-friendly

### Intermediate Level

1. **Add Image Preview Enhancements**
   - File: `frontend/src/components/ImageUpload.jsx`
   - Show image dimensions
   - Display file size
   - Add image filters

2. **Implement Prediction History**
   - Add state to store past predictions
   - Display history in UI
   - Allow clearing history

3. **Add Multiple Image Upload**
   - Modify upload component
   - Process multiple images
   - Display batch results

### Advanced Level

1. **Add Database Integration**
   - Install PostgreSQL/MongoDB
   - Store predictions
   - Add user accounts

2. **Implement Different Models**
   - File: `backend/app/model.py`
   - Add ResNet50 or EfficientNet
   - Allow model selection in UI

3. **Add Real-time Processing**
   - Implement WebSockets
   - Stream predictions
   - Show progress updates

4. **Custom Model Training**
   - Train on custom dataset
   - Replace MobileNetV2
   - Deploy your model

## 🔧 Making Changes

### Adding a New API Endpoint

**Backend** (`backend/app/main.py`):

```python
@app.get("/your-endpoint")
async def your_function():
    """
    Your endpoint description
    """
    return {"message": "Hello"}
```

**Frontend** (`frontend/src/services/api.js`):

```javascript
export async function yourFunction() {
  const response = await apiClient.get('/your-endpoint')
  return response.data
}
```

**Usage** (`frontend/src/App.jsx`):

```javascript
import { yourFunction } from './services/api'

const data = await yourFunction()
```

### Adding a New Component

**Create Component** (`frontend/src/components/NewComponent.jsx`):

```javascript
function NewComponent({ prop1, prop2 }) {
  return (
    <div className="new-component">
      <h2>{prop1}</h2>
      <p>{prop2}</p>
    </div>
  )
}

export default NewComponent
```

**Use Component** (`frontend/src/App.jsx`):

```javascript
import NewComponent from './components/NewComponent'

// In render:
<NewComponent prop1="Hello" prop2="World" />
```

**Add Styles** (`frontend/src/App.css`):

```css
.new-component {
  padding: 1rem;
  background: white;
}
```

### Modifying the ML Model

**Option 1: Change Existing Model**

```python
# backend/app/model.py

# Instead of MobileNetV2
from tensorflow.keras.applications import ResNet50

self.model = ResNet50(
    include_top=True,
    weights='imagenet',
    input_shape=self.input_shape
)
```

**Option 2: Add Model Selection**

```python
# backend/app/model.py

class ModelFactory:
    @staticmethod
    def create_model(model_name: str):
        if model_name == "mobilenet":
            return MobileNetV2(...)
        elif model_name == "resnet":
            return ResNet50(...)
        # Add more models
```

## 🧪 Testing Your Changes

### Test Backend Changes

```bash
cd backend
source venv/bin/activate

# Test specific endpoint
curl http://localhost:8000/your-endpoint

# Run tests
pytest tests/ -v
```

### Test Frontend Changes

```bash
cd frontend

# Development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## 📝 Code Style Guidelines

### Python (Backend)

- Use **docstrings** for all functions
- Follow **PEP 8** style guide
- Add **type hints** where possible
- Write **descriptive variable names**

Example:
```python
def process_image(image: np.ndarray) -> List[Dict]:
    """
    Process and classify an image.

    Args:
        image: numpy array of the image

    Returns:
        List of prediction dictionaries
    """
    # Implementation
    pass
```

### JavaScript (Frontend)

- Use **const/let** instead of var
- Write **clear function names**
- Add **comments** for complex logic
- Keep components **small and focused**

Example:
```javascript
/**
 * Classify an image using the ML model
 * @param {File} imageFile - The image to classify
 * @returns {Promise<Object>} Prediction results
 */
async function classifyImage(imageFile) {
  // Implementation
}
```

## 🐛 Debugging Tips

### Backend Debugging

1. **Add Logging**
   ```python
   import logging
   logger = logging.getLogger(__name__)
   logger.info(f"Processing image: {filename}")
   ```

2. **Use Debugger**
   ```python
   import pdb; pdb.set_trace()  # Add breakpoint
   ```

3. **Check Logs**
   ```bash
   tail -f logs/app.log
   ```

### Frontend Debugging

1. **Console Logging**
   ```javascript
   console.log('Image selected:', file)
   console.error('API call failed:', error)
   ```

2. **React DevTools**
   - Install React DevTools browser extension
   - Inspect component state and props

3. **Network Tab**
   - F12 > Network
   - See all API requests
   - Check request/response data

## 📦 Adding Dependencies

### Backend (Python)

```bash
cd backend
source venv/bin/activate

# Install new package
pip install package-name

# Update requirements
pip freeze > requirements.txt
```

### Frontend (Node)

```bash
cd frontend

# Install new package
npm install package-name

# Dev dependency
npm install --save-dev package-name
```

## 🚀 Submitting Your Work

### For Course Assignments

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code
   - Test thoroughly
   - Add comments

3. **Commit Changes**
   ```bash
   git add .
   git commit -m "Add: descriptive message"
   ```

4. **Push to GitHub**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Deploy**
   - Follow DEPLOYMENT.md
   - Share deployment URLs

6. **Document**
   - Update README if needed
   - Add screenshots
   - Explain your changes

### Commit Message Guidelines

- **Add**: New feature
- **Fix**: Bug fix
- **Update**: Modify existing feature
- **Refactor**: Code improvement
- **Docs**: Documentation changes

Examples:
```
Add: Image compression before upload
Fix: CORS error in production
Update: Improve prediction accuracy
Refactor: Simplify API error handling
Docs: Add deployment troubleshooting section
```

## 💡 Ideas for Extensions

### UI/UX Improvements
- Dark mode toggle
- Responsive mobile design
- Drag & drop multiple files
- Animated transitions
- Progressive Web App (PWA)

### Features
- Batch image processing
- Export predictions as CSV/JSON
- Share results via link
- Comparison mode (multiple images)
- Custom model upload

### Technical
- Add caching (Redis)
- Implement rate limiting
- Add user authentication
- WebSocket for real-time updates
- Containerize with Docker

### ML Enhancements
- Object detection (YOLO)
- Image segmentation
- Multiple models comparison
- Confidence threshold adjustment
- Custom class labels

## 📚 Resources for Learning

### React
- [Official React Docs](https://react.dev/)
- [React Tutorial](https://react.dev/learn)
- [React Hooks Guide](https://react.dev/reference/react)

### FastAPI
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [Pydantic Models](https://docs.pydantic.dev/)

### TensorFlow
- [TensorFlow Guides](https://www.tensorflow.org/guide)
- [Keras Applications](https://keras.io/api/applications/)
- [Transfer Learning](https://www.tensorflow.org/tutorials/images/transfer_learning)

### Deployment
- [Vercel Docs](https://vercel.com/docs)
- [Railway Docs](https://docs.railway.app/)
- [Docker Tutorial](https://docs.docker.com/get-started/)

## ❓ Getting Help

1. **Read the Documentation**
   - README.md
   - TESTING.md
   - DEPLOYMENT.md

2. **Check Logs**
   - Backend: Console output
   - Frontend: Browser console (F12)

3. **Search Online**
   - Stack Overflow
   - GitHub Issues
   - Official documentation

4. **Ask Your Instructor**
   - Provide error messages
   - Share code snippets
   - Explain what you've tried

## 🎯 Best Practices

1. **Always test locally** before deploying
2. **Write clean, commented code**
3. **Use version control** (git)
4. **Keep dependencies updated**
5. **Document your changes**
6. **Handle errors gracefully**
7. **Think about security**
8. **Optimize performance**

---

**Happy Learning!** 🚀

Remember: The best way to learn is by doing. Don't be afraid to experiment, break things, and fix them. That's how you learn!
