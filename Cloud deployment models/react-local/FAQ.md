# Frequently Asked Questions (FAQ)

Common questions students ask about this React + Deep Learning tutorial.

---

## General Questions

### Q1: Do I need Python or a GPU to run this?
**A:** No! Everything runs in your browser using JavaScript. TensorFlow.js uses WebGL for GPU acceleration automatically if available, but it works fine on CPU too.

### Q2: How long does it take to train the model?
**A:** Typically 30-60 seconds with the default settings (5000 training images, 10 epochs). This depends on your computer's speed and browser.

### Q3: Can I use this offline?
**A:** After the first run (which downloads TensorFlow.js), you can work offline during development. For production, you'd need to bundle all dependencies.

### Q4: Is this real machine learning?
**A:** Yes! This is a real Convolutional Neural Network that learns from data. The only simplification is using synthetic training data instead of the full MNIST dataset for faster downloads.

---

## React Questions

### Q5: What is `useState` and why do we use it?
**A:** `useState` is a React Hook that lets components remember values between renders. For example:

```javascript
const [count, setCount] = useState(0);
// count is the current value (0 initially)
// setCount is a function to update it
```

When you call `setCount(5)`, React re-renders the component with the new value.

### Q6: What is `useEffect` for?
**A:** `useEffect` runs code after your component renders. Common uses:
- Fetching data when component mounts
- Setting up event listeners
- Cleaning up resources

```javascript
useEffect(() => {
  // This runs after first render
  console.log('Component mounted!');
}, []); // Empty array = run once
```

### Q7: What is `useRef` and why use it instead of `useState` for the canvas?
**A:** `useRef` gives you a direct reference to a DOM element. Unlike `useState`, updating a ref doesn't trigger a re-render. Perfect for:
- Accessing DOM elements (like canvas)
- Storing values that shouldn't trigger re-renders

```javascript
const canvasRef = useRef(null);
// Later: canvasRef.current gives you the actual canvas element
```

### Q8: Why do we pass functions as props (onModelReady, onCanvasChange)?
**A:** This is how child components communicate with parent components in React. The child "calls back" to the parent when something happens:

```javascript
// Parent (App.js)
<DrawingCanvas onCanvasChange={handleCanvasChange} />

// Child (DrawingCanvas.js)
if (onCanvasChange) {
  onCanvasChange(canvas); // Calls parent's function
}
```

---

## TensorFlow.js Questions

### Q9: What is a tensor?
**A:** A tensor is a multi-dimensional array. Think of it like:
- 1D tensor = array: `[1, 2, 3]`
- 2D tensor = matrix: `[[1, 2], [3, 4]]`
- 3D tensor = cube of numbers (like an image with width, height, colors)

In our app, images are 3D tensors: `[28, 28, 1]` = 28 pixels wide, 28 pixels tall, 1 color channel.

### Q10: Why do we need to dispose tensors?
**A:** Unlike regular JavaScript variables, TensorFlow.js tensors are stored in GPU memory and aren't automatically garbage collected. You must manually dispose them:

```javascript
const tensor = tf.tensor([1, 2, 3]);
tensor.dispose(); // Free memory

// Or use tf.tidy() for automatic cleanup:
const result = tf.tidy(() => {
  const a = tf.tensor([1, 2]);
  const b = tf.tensor([3, 4]);
  return a.add(b); // Only the result survives
}); // a and b are auto-disposed
```

### Q11: What's the difference between model.fit() and model.predict()?
**A:**
- **model.fit()**: Training - the model learns from data
- **model.predict()**: Inference - using the trained model to make predictions

```javascript
// Training (learning from examples)
await model.fit(images, labels);

// Prediction (using what it learned)
const prediction = model.predict(newImage);
```

### Q12: What does sequential() mean?
**A:** `tf.sequential()` creates a model where layers are stacked linearly (one after another). Data flows through layers in sequence:

```
Input → Layer1 → Layer2 → Layer3 → Output
```

For more complex architectures (like branching), you'd use the functional API.

---

## Deep Learning Questions

### Q13: What is a Convolutional Neural Network (CNN)?
**A:** A CNN is a type of neural network designed for image processing. It has special layers:
- **Convolutional layers**: Detect features (edges, shapes, patterns)
- **Pooling layers**: Reduce size while keeping important info
- **Dense layers**: Make final classification decision

CNNs are great for images because they learn spatial patterns (like "vertical line at position X").

### Q14: What is ReLU activation?
**A:** ReLU (Rectified Linear Unit) is a simple function:
```
ReLU(x) = max(0, x)
```
- If input is positive, output = input
- If input is negative, output = 0

It introduces non-linearity, allowing the network to learn complex patterns.

### Q15: What is Softmax activation?
**A:** Softmax converts raw scores into probabilities that sum to 1:

```
Input:  [2.3, 1.2, 0.5, 4.1, ...]
Softmax: [0.12, 0.04, 0.02, 0.72, ...] // sums to 1.0
```

Perfect for classification - each output represents the probability of that class.

### Q16: What is the difference between loss and accuracy?
**A:**
- **Loss**: How wrong predictions are (continuous value)
  - Lower is better
  - 0 = perfect predictions
  - Optimized directly during training

- **Accuracy**: Percentage of correct predictions
  - Higher is better
  - 1.0 (100%) = all correct
  - Easy to understand

Both metrics are useful - loss for optimization, accuracy for evaluation.

### Q17: What is an epoch?
**A:** One complete pass through the entire training dataset. If you have 5000 images and train for 10 epochs, the model sees each image 10 times (not all at once, but in batches).

### Q18: What is batch size?
**A:** Number of training examples processed together before updating model weights. In our app, batch size = 64:
- Process 64 images
- Calculate average error
- Update weights
- Repeat for next 64 images

Larger batches = faster training but more memory usage.

### Q19: Why do we need training AND validation data?
**A:**
- **Training data**: Model learns from this
- **Validation data**: Tests how well model generalizes to new data

If validation accuracy is much lower than training accuracy → overfitting (model memorized training data instead of learning patterns).

---

## Canvas Questions

### Q20: How does the canvas drawing work?
**A:** The Canvas API provides a 2D drawing context:

```javascript
const canvas = document.getElementById('myCanvas');
const ctx = canvas.getContext('2d');

// Draw a line
ctx.beginPath();
ctx.moveTo(x1, y1); // Start point
ctx.lineTo(x2, y2); // End point
ctx.stroke();       // Actually draw it
```

We track mouse position and draw lines between points.

### Q21: How do we convert canvas to tensor?
**A:**
```javascript
// Get canvas pixel data
const ctx = canvas.getContext('2d');
const imageData = ctx.getImageData(0, 0, 280, 280);

// Convert to tensor
let tensor = tf.browser.fromPixels(imageData, 1); // 1 = grayscale

// Resize to 28x28 (MNIST size)
tensor = tf.image.resizeBilinear(tensor, [28, 28]);

// Normalize to [0, 1]
tensor = tensor.div(255.0);
```

---

## Performance Questions

### Q22: Why is my app slow?
**A:** Several possible causes:
1. **Large dataset**: Reduce `trainDataSize` in `modelUtils.js`
2. **Too many epochs**: Reduce from 10 to 5
3. **CPU training**: TensorFlow.js prefers Chrome for best WebGL support
4. **Memory leaks**: Make sure tensors are disposed

### Q23: How can I make training faster?
**A:**
1. Use Chrome browser (best WebGL support)
2. Reduce training data size:
   ```javascript
   const trainDataSize = 1000; // Instead of 5000
   ```
3. Reduce epochs:
   ```javascript
   const epochs = 5; // Instead of 10
   ```
4. Increase batch size:
   ```javascript
   const batchSize = 128; // Instead of 64
   ```

### Q24: How much memory does this use?
**A:** Typically 100-300 MB, depending on:
- Model size
- Training data size
- Number of tensors in memory

Check memory usage:
```javascript
console.log(tf.memory());
```

---

## Debugging Questions

### Q25: How do I see what's happening?
**A:** Use browser console (F12 → Console tab):
```javascript
console.log('Current prediction:', prediction);
console.log('Model summary:', model.summary());
console.log('Memory:', tf.memory());
```

### Q26: My model always predicts the same digit!
**A:** Possible causes:
1. **Model not trained**: Click "Train Model" and wait
2. **Drawing is unclear**: Draw larger, clearer digits
3. **Model overfit**: Try retraining with different parameters

### Q27: I get "tensor is disposed" errors
**A:** You're trying to use a tensor after it's been disposed:
```javascript
const tensor = tf.tensor([1, 2, 3]);
tensor.dispose();
tensor.print(); // ERROR: tensor is disposed

// Fix: Don't dispose until you're done using it
```

### Q28: The app crashes with "out of memory"
**A:** Tensors aren't being disposed properly:
```javascript
// Bad:
for (let i = 0; i < 1000; i++) {
  const t = tf.tensor([i]); // Creates 1000 tensors in memory!
}

// Good:
for (let i = 0; i < 1000; i++) {
  const t = tf.tensor([i]);
  // Use t...
  t.dispose(); // Clean up immediately
}

// Better:
tf.tidy(() => {
  for (let i = 0; i < 1000; i++) {
    const t = tf.tensor([i]);
    // Use t... (auto-disposed at end of tidy)
  }
});
```

---

## Conceptual Questions

### Q29: How does the model "learn"?
**A:** Through a process called backpropagation:
1. Model makes prediction (forward pass)
2. Compare prediction to correct answer (calculate loss)
3. Calculate how to adjust each weight to reduce loss (backward pass)
4. Update weights slightly in that direction
5. Repeat thousands of times

Over time, the model gets better at making correct predictions.

### Q30: Can the model learn to recognize my handwriting specifically?
**A:** Not with this tutorial's setup, but you could:
1. Draw 100+ examples of each digit (0-9) in your handwriting
2. Label them
3. Train a new model on your data
4. Save the model for future use

This is called "fine-tuning" or "transfer learning".

### Q31: What makes a good training dataset?
**A:**
- **Large enough**: More data = better learning (usually thousands of examples)
- **Balanced**: Equal numbers of each class
- **Diverse**: Different styles, sizes, positions
- **Accurate labels**: Correct answers are crucial
- **Representative**: Similar to what you'll predict on

---

## Extension Questions

### Q32: How can I use this model in production?
**A:**
1. Train the model
2. Save it: `await model.save('downloads://my-model')`
3. In production, load it: `await tf.loadLayersModel('path/to/model.json')`
4. Use for predictions without retraining

### Q33: Can I use a pre-trained model instead?
**A:** Yes! TensorFlow.js Hub has many pre-trained models:
```javascript
import * as mobilenet from '@tensorflow-models/mobilenet';

const model = await mobilenet.load();
const predictions = await model.classify(image);
```

### Q34: How do I deploy this app?
**A:**
```bash
npm run build
```
This creates a `build/` folder with optimized files. Deploy to:
- GitHub Pages
- Netlify
- Vercel
- Any static hosting service

---

## Still Have Questions?

1. **Check the code comments**: Every file has detailed explanations
2. **Read TUTORIAL.md**: Step-by-step walkthrough
3. **Experiment**: Try changing things and see what happens
4. **Use the console**: `console.log()` is your friend
5. **Google it**: Many others have had similar questions

**Remember**: The best way to learn is by doing! 🚀
