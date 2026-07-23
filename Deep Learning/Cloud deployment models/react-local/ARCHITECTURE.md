# Architecture Overview

This document explains the technical architecture of the React Deep Learning tutorial app.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Browser (Client)                         │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    React Application                        │ │
│  │                                                              │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │ │
│  │  │ App.js       │  │ Components   │  │ Utils            │ │ │
│  │  │              │  │              │  │                  │ │ │
│  │  │ - Main state │  │ - DrawingCanvas│ │ - modelUtils.js│ │ │
│  │  │ - Orchestration│ │ - ModelTrainer│ │   (TensorFlow) │ │ │
│  │  │              │  │ - PredictionDisplay│               │ │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘ │ │
│  │                                                              │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    TensorFlow.js                            │ │
│  │                                                              │ │
│  │  - Neural Network Model (CNN)                               │ │
│  │  - Training Engine (Backpropagation)                        │ │
│  │  - WebGL Backend (GPU Acceleration)                         │ │
│  │  - Memory Management                                        │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                     Browser APIs                            │ │
│  │                                                              │ │
│  │  - Canvas API (Drawing)                                     │ │
│  │  - WebGL (GPU Computation)                                  │ │
│  │  - LocalStorage (Model Persistence - optional)              │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Key Point**: Everything runs locally in the browser - no backend server required!

---

## Component Architecture

```
App.js (Root Component)
├── State Management
│   ├── model: Trained TensorFlow.js model
│   └── prediction: Current prediction results
│
├── Event Handlers
│   ├── handleModelReady(): Called when training completes
│   └── handleCanvasChange(): Called when user draws
│
└── Child Components
    │
    ├── ModelTrainer
    │   ├── Props: onModelReady (callback)
    │   ├── State: model, data, status, progress
    │   └── Actions: Create model, load data, train model
    │
    ├── DrawingCanvas
    │   ├── Props: onCanvasChange (callback)
    │   ├── Refs: canvasRef (DOM reference)
    │   └── Actions: Draw, clear, capture image data
    │
    └── PredictionDisplay
        ├── Props: prediction (results)
        └── Actions: Display digit, confidence, probabilities
```

---

## Data Flow Diagram

### 1. Training Flow

```
User clicks "Train Model"
    ↓
ModelTrainer.handleTrainModel()
    ↓
modelUtils.createModel()
    ↓ (creates CNN)
modelUtils.loadMNISTData()
    ↓ (loads training images)
modelUtils.trainModel()
    ├─→ Forward pass (predict)
    ├─→ Calculate loss
    ├─→ Backward pass (update weights)
    └─→ Repeat for each epoch
    ↓
onModelReady(trainedModel)
    ↓
App.setModel(trainedModel)
    ↓
Model ready for predictions!
```

### 2. Prediction Flow

```
User draws on canvas
    ↓
DrawingCanvas detects mouse movement
    ↓
onCanvasChange(canvas)
    ↓
App.handleCanvasChange(canvas)
    ↓
modelUtils.preprocessCanvas(canvas)
    ├─→ Extract image data
    ├─→ Resize to 28x28
    ├─→ Convert to grayscale
    └─→ Normalize to [0, 1]
    ↓
modelUtils.predict(model, tensor)
    ├─→ Run forward pass through model
    ├─→ Get probabilities for each digit
    └─→ Find highest probability
    ↓
App.setPrediction(results)
    ↓
PredictionDisplay shows results
```

---

## Neural Network Architecture

```
Input: 28x28x1 Grayscale Image
    ↓
┌─────────────────────────────────────┐
│ Conv2D Layer 1                      │
│ - 8 filters, 5x5 kernel             │
│ - ReLU activation                   │
│ - Output: 24x24x8                   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ MaxPooling2D Layer 1                │
│ - Pool size: 2x2                    │
│ - Output: 12x12x8                   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Conv2D Layer 2                      │
│ - 16 filters, 5x5 kernel            │
│ - ReLU activation                   │
│ - Output: 8x8x16                    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ MaxPooling2D Layer 2                │
│ - Pool size: 2x2                    │
│ - Output: 4x4x16                    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Flatten Layer                       │
│ - Converts 4x4x16 to 256            │
│ - Output: 256                       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Dense Layer                         │
│ - 128 units                         │
│ - ReLU activation                   │
│ - Output: 128                       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Output Dense Layer                  │
│ - 10 units (one per digit 0-9)      │
│ - Softmax activation                │
│ - Output: 10 probabilities          │
└─────────────────────────────────────┘
    ↓
Output: [p0, p1, p2, ..., p9]
(Probabilities for each digit)
```

### Parameter Count

```
Layer               Output Shape      Params
─────────────────────────────────────────────
Conv2D_1           [24, 24, 8]       208
MaxPool2D_1        [12, 12, 8]       0
Conv2D_2           [8, 8, 16]        3,216
MaxPool2D_2        [4, 4, 16]        0
Flatten            [256]             0
Dense_1            [128]             32,896
Dense_2 (Output)   [10]              1,290
─────────────────────────────────────────────
Total params: 37,610
Trainable params: 37,610
```

---

## File Structure & Responsibilities

```
src/
│
├── index.js
│   └── Renders App component into DOM
│
├── App.js
│   ├── Main orchestration
│   ├── Manages global state (model, predictions)
│   └── Coordinates child components
│
├── App.css
│   └── All styling (CSS variables, responsive design)
│
├── components/
│   │
│   ├── DrawingCanvas.js
│   │   ├── Renders HTML5 canvas
│   │   ├── Handles mouse/touch events
│   │   ├── Manages drawing state
│   │   └── Captures image data
│   │
│   ├── ModelTrainer.js
│   │   ├── Creates TensorFlow.js model
│   │   ├── Loads training data
│   │   ├── Manages training process
│   │   ├── Displays training metrics
│   │   └── Notifies parent when training completes
│   │
│   └── PredictionDisplay.js
│       ├── Receives prediction results as props
│       ├── Displays predicted digit
│       ├── Shows confidence score
│       └── Renders probability distribution
│
└── utils/
    │
    └── modelUtils.js
        ├── createModel(): Builds CNN architecture
        ├── loadMNISTData(): Loads/generates training data
        ├── trainModel(): Training loop with callbacks
        ├── preprocessCanvas(): Image preprocessing
        ├── predict(): Makes predictions
        └── Helper functions (memory info, etc.)
```

---

## State Management

### App.js (Parent State)

```javascript
const [model, setModel] = useState(null);
// - Holds the trained TensorFlow.js model
// - null initially, set after training
// - Shared with all child components via callbacks

const [prediction, setPrediction] = useState(null);
// - Holds current prediction results
// - Updated on every canvas change
// - Passed to PredictionDisplay as prop
```

### ModelTrainer.js (Local State)

```javascript
const [model, setModel] = useState(null);
// - Local model reference during training
// - Passed to parent via callback when ready

const [data, setData] = useState(null);
// - Training and validation datasets
// - Loaded once on component mount

const [status, setStatus] = useState('idle');
// - Training status: idle, loading, ready, training, trained
// - Controls UI state (button disabled, progress bar, etc.)

const [progress, setProgress] = useState({...});
// - Training metrics (epoch, loss, accuracy)
// - Updated during training via callbacks
```

### DrawingCanvas.js (Local State)

```javascript
const canvasRef = useRef(null);
// - Reference to canvas DOM element
// - Used for drawing operations

const [isDrawing, setIsDrawing] = useState(false);
// - Tracks whether user is currently drawing
// - Controls whether mouse movement draws lines
```

---

## Event Flow

### User Interaction → State Update → UI Re-render

```
User Action                  Event Handler              State Update         UI Update
────────────────────────────────────────────────────────────────────────────────────────
Click "Train Model"    →    handleTrainModel()    →    setStatus()     →    Button disabled
                                                        setProgress()   →    Progress bar
                                                        setModel()      →    Badge: "Trained"

Draw on canvas         →    handleCanvasChange()  →    setPrediction() →    Display updated

Clear canvas           →    clearCanvas()         →    setPrediction() →    Display cleared
```

---

## Memory Management

TensorFlow.js requires explicit memory management to prevent leaks.

### Tensor Lifecycle

```javascript
// CREATION
const tensor = tf.tensor([1, 2, 3]);
// ↓ Allocates GPU memory

// USAGE
const result = tensor.mul(2);
// ↓ Creates new tensor, original still in memory

// DISPOSAL
tensor.dispose();
result.dispose();
// ↓ Frees GPU memory
```

### Using tf.tidy() for Auto-cleanup

```javascript
const result = tf.tidy(() => {
  const a = tf.tensor([1, 2, 3]);
  const b = tf.tensor([4, 5, 6]);
  const sum = a.add(b);
  return sum; // Only sum survives
}); // a and b are auto-disposed

// Later:
result.dispose(); // Clean up when done
```

### Where We Use Memory Management

1. **preprocessCanvas()**:
   ```javascript
   return tf.tidy(() => {
     // All intermediate tensors auto-cleaned
   });
   ```

2. **predict()**:
   ```javascript
   return tf.tidy(() => {
     // Prediction tensors auto-cleaned
   });
   ```

3. **trainModel()**:
   ```javascript
   // Manual disposal after training
   xsTensor.dispose();
   ysTensor.dispose();
   ```

---

## Performance Optimizations

### 1. WebGL Backend
TensorFlow.js automatically uses WebGL for GPU acceleration when available.

### 2. Batch Processing
Training processes 64 images at a time for efficiency.

### 3. Data Pipeline
Uses `tf.data.Dataset` API for efficient data loading.

### 4. Lazy Loading
Only loads TensorFlow.js when needed (not blocking initial page load).

### 5. Memory Management
Aggressive tensor disposal prevents memory leaks.

---

## Browser Compatibility

### Supported Browsers

| Browser | Version | WebGL | Performance |
|---------|---------|-------|-------------|
| Chrome  | 60+     | ✅    | Excellent   |
| Firefox | 55+     | ✅    | Good        |
| Safari  | 11+     | ✅    | Good        |
| Edge    | 79+     | ✅    | Excellent   |

### Fallback Behavior

If WebGL is unavailable, TensorFlow.js falls back to:
1. WASM backend (slower but works)
2. CPU backend (slowest but always works)

---

## Security Considerations

### No Backend = Enhanced Privacy
- All data stays in user's browser
- No data sent to servers
- No API keys or authentication needed

### Potential Concerns
- Model could be extracted from browser
- Training data visible in network tab
- No server-side validation

**For Production**: Consider server-side validation for sensitive applications.

---

## Scalability

### Current Limitations
- Training on ~5000 examples (small dataset)
- Simple CNN architecture
- Single-digit recognition only

### How to Scale

1. **Larger Datasets**:
   ```javascript
   const trainDataSize = 50000; // Full MNIST
   ```

2. **Deeper Models**:
   ```javascript
   // Add more layers
   model.add(tf.layers.conv2d({filters: 32, ...}));
   model.add(tf.layers.conv2d({filters: 64, ...}));
   ```

3. **Transfer Learning**:
   ```javascript
   // Use pre-trained model as base
   const baseModel = await tf.loadLayersModel('...');
   // Add custom layers on top
   ```

4. **Web Workers**:
   ```javascript
   // Run training in background thread
   const worker = new Worker('training-worker.js');
   ```

---

## Testing Strategy

### Unit Tests (Not included, but recommended)

```javascript
// Test model creation
test('createModel returns valid model', () => {
  const model = createModel();
  expect(model.layers.length).toBeGreaterThan(0);
});

// Test preprocessing
test('preprocessCanvas resizes to 28x28', () => {
  const tensor = preprocessCanvas(mockCanvas);
  expect(tensor.shape).toEqual([1, 28, 28, 1]);
});
```

### Integration Tests

```javascript
// Test full prediction pipeline
test('end-to-end prediction', async () => {
  const model = createModel();
  await trainModel(model, ...);
  const result = predict(model, testImage);
  expect(result.predictedDigit).toBeGreaterThanOrEqual(0);
  expect(result.predictedDigit).toBeLessThan(10);
});
```

---

## Deployment Options

### 1. Static Hosting (Simplest)

```bash
npm run build
# Deploy /build folder to:
# - GitHub Pages
# - Netlify
# - Vercel
# - AWS S3
```

### 2. Docker Container

```dockerfile
FROM node:16-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
RUN npm install -g serve
CMD ["serve", "-s", "build", "-l", "3000"]
```

### 3. Progressive Web App (PWA)

Add service worker for offline capability:
```javascript
// Enable in src/index.js
serviceWorker.register();
```

---

## Future Enhancements

### Potential Features

1. **Model Persistence**:
   ```javascript
   await model.save('localstorage://my-model');
   ```

2. **Real MNIST Data**:
   ```javascript
   import {MnistData} from './mnist-data.js';
   ```

3. **Webcam Integration**:
   ```javascript
   const video = await tf.data.webcam(videoElement);
   ```

4. **Multi-Model Support**:
   - Switch between architectures
   - Compare performance

5. **Visualization**:
   - Layer activation visualization
   - Training curve plots
   - Confusion matrix

---

## Conclusion

This architecture demonstrates:
✅ Modern React patterns (hooks, component composition)
✅ Browser-based machine learning (TensorFlow.js)
✅ Real-time inference (interactive predictions)
✅ Educational design (extensive comments, clear structure)

Perfect for learning both React and Deep Learning! 🚀
