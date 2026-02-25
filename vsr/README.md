# WebNN Sample App

A browser-based ONNX model inference application using **ONNX Runtime Web** with **WebNN** (Web Neural Network) Execution Provider support.

This is similar to `WinMLSampleApp` but runs entirely in the browser with NPU/GPU/CPU acceleration via WebNN.

---

## 📋 Version Information

| Component | Version | Notes |
|-----------|---------|-------|
| **ONNX Runtime Web** | 1.24.1 | Loaded via CDN |
| **WebNN API** | Chrome 120+ | Experimental, requires flag |
| **Tested Browser** | Chrome/Edge | February 2026 |
| **Tested OS** | Windows 11 | With NPU support |

---

## 🚀 Quick Start

### Step 1: Start a Local Server

A local HTTP server is required because browsers block loading local files directly.

**Using Python:**
```bash
cd e:\pst\webnn\WebNNSampleApp
python -m http.server 8080
```

**Using Node.js:**
```bash
cd e:\pst\webnn\WebNNSampleApp
npx http-server -p 8080 -c-1 -a 127.0.0.1
```

### Step 2: Enable WebNN in Browser

WebNN is an experimental API that requires a browser flag:

1. Open Chrome or Edge
2. Navigate to: `chrome://flags/#web-machine-learning-neural-network`
3. Set to **"Enabled"**
4. Click **"Relaunch"** to restart the browser

### Step 3: Open the App

1. Navigate to: `http://localhost:8080`
2. You should see "WebNN API is available" in the log

### Step 4: Load and Run a Model

1. Select Execution Provider: **WebNN**
2. Select Device Type: **NPU** (or GPU/CPU)
3. Click "Choose File" and select an ONNX model
4. Click **"Load Model"**
5. Set the correct **Input Shape** (e.g., `1,3,400,400`)
6. Set the correct **Input Data Type** (e.g., `float32`)
7. Click **"Run with Random Input"**

---

## 📁 Files

| File | Description |
|------|-------------|
| `index.html` | Main HTML page with UI layout and styles |
| `main.js` | Application logic (heavily commented for learning) |
| `README.md` | This documentation file |

---

## ⚙️ Execution Providers

| EP | Backend | Best For | Requirements |
|----|---------|----------|--------------|
| **WebNN** | NPU/GPU/CPU via Web Neural Network API | Power efficiency, NPU acceleration | Chrome/Edge + flag enabled |
| **WebGPU** | GPU via WebGPU shaders | GPU compute, good compatibility | Chrome 113+ |
| **WASM** | CPU via WebAssembly | Maximum compatibility | All modern browsers |

### WebNN Device Types

| Device | Description |
|--------|-------------|
| `cpu` | Software fallback (always available) |
| `gpu` | GPU via DirectML (Windows) or Metal (macOS) |
| `npu` | Neural Processing Unit (if available) |

### WebNN Power Preferences

| Mode | Description |
|------|-------------|
| `default` | System decides |
| `low-power` | Prefer integrated GPU, optimize for battery |
| `high-performance` | Prefer discrete GPU, optimize for speed |

---

## ⚠️ Model Requirements

### ✅ Supported Models (for WebNN)

- **Float32 models** - Full support
- **Float16 models** - Good support
- Simple quantized models with per-tensor quantization

### ❌ Known Limitations (for WebNN)

| Issue | Error Message | Solution |
|-------|---------------|----------|
| Per-channel quantization | `dequantizeLinear: scale rank != input rank` | Use float32 model or WASM EP |
| Unsupported operators | `Unsupported operator: XYZ` | Use WASM EP for full coverage |
| Large/complex models | Hangs during load | Try WASM EP first to verify model works |

### Quantization Explained

ONNX models can be quantized (compressed) to reduce size and improve speed:

- **Per-tensor quantization**: One scale value for entire tensor ✅ WebNN OK
- **Per-channel quantization**: One scale per channel ❌ WebNN fails

If your quantized model fails with WebNN, use:
1. **WASM EP** - Works with all quantization types
2. **Float32 model** - No quantization needed

---

## 🔧 Troubleshooting

### "WebNN API is NOT available"

**Solution:**
1. Use Chrome or Edge (not Firefox/Safari)
2. Enable flag: `chrome://flags/#web-machine-learning-neural-network`
3. Restart browser completely

### Model loads but inference hangs

**Diagnosis:**
1. Click **"Test WebNN Directly"** to test WebNN API
2. If direct test works but model fails, model has unsupported ops
3. Try **WASM EP** to verify model works

**Solution:**
- Use float32 version of the model
- Use WASM EP for complex models

### "Unexpected input data type"

**Example:** `Actual: tensor(float), expected: tensor(uint8)`

**Solution:**
1. Check your model's expected input type (use Netron: https://netron.app)
2. Set correct **Input Data Type** dropdown (uint8, float32, etc.)

### "Invalid input shape"

**Solution:**
1. Check your model's expected input shape (use Netron)
2. Enter correct shape in **Input Shape** field
3. Format: comma-separated, e.g., `1,3,400,400`

---

## 📊 Common Input Shapes

| Model Type | Typical Shape | Format |
|------------|---------------|--------|
| Image Classification | `1,3,224,224` | NCHW |
| Image Classification | `1,224,224,3` | NHWC |
| Streaming SR (small) | `1,256,288,3` | NHWC |
| Streaming SR (large) | `1,736,864,3` | NHWC |
| Custom | Check model with Netron | - |

**Shape Format:**
- `N` = Batch size (usually 1)
- `C` = Channels (3 for RGB)
- `H` = Height
- `W` = Width

---

## 🔍 Debugging Tools

### Test WebNN Directly

The **"Test WebNN Directly"** button runs a minimal WebNN graph (`y = x * 2`) without ONNX Runtime. This helps isolate issues:

| Direct Test | ONNX Model | Issue Is With |
|-------------|------------|---------------|
| ✅ Pass | ✅ Pass | Everything works! |
| ✅ Pass | ❌ Fail | Model or ONNX Runtime |
| ❌ Fail | ❌ Fail | WebNN setup/browser |

### Browser DevTools

Press **F12** to open DevTools:
- **Console tab**: See JavaScript errors
- **Network tab**: Check if files load correctly
- **Sources tab**: Debug main.js

### Netron Model Viewer

Use https://netron.app to inspect ONNX models:
- See input/output shapes and types
- View all operators used
- Identify problematic ops

---

## 📚 References

### ONNX Runtime Web
- Documentation: https://onnxruntime.ai/docs/get-started/with-javascript/web.html
- GitHub: https://github.com/microsoft/onnxruntime
- NPM: https://www.npmjs.com/package/onnxruntime-web

### WebNN API
- W3C Spec: https://www.w3.org/TR/webnn/
- Chrome Status: https://chromestatus.com/feature/5738583861886976
- WebNN Samples: https://webmachinelearning.github.io/webnn-samples/

### ONNX
- ONNX Models: https://github.com/onnx/models
- Model Zoo: https://onnx.ai/models/
- Netron Viewer: https://netron.app

---

## 📝 Code Overview

### Key Functions in main.js

| Function | Purpose |
|----------|---------|
| `checkWebNNSupport()` | Checks if WebNN API is available |
| `testWebNNDirect()` | Tests WebNN without ONNX Runtime |
| `getSessionOptions()` | Builds EP configuration for ONNX Runtime |
| `loadModel()` | Loads ONNX model and creates InferenceSession |
| `createRandomTensor()` | Creates test tensor with random data |
| `runInference()` | Runs model inference once |
| `runBenchmark()` | Runs multiple iterations for performance testing |

### ONNX Runtime Web API

```javascript
// Load model
const session = await ort.InferenceSession.create(modelData, {
    executionProviders: [{
        name: 'webnn',
        deviceType: 'npu'  // 'cpu', 'gpu', or 'npu'
    }]
});

// Create input tensor
const inputTensor = new ort.Tensor('float32', data, [1, 3, 224, 224]);

// Run inference
const results = await session.run({ 'input': inputTensor });

// Get output
const outputData = results['output'].data;
```

### WebNN API (Direct)

```javascript
// Create context for NPU
const context = await navigator.ml.createContext({ deviceType: 'npu' });

// Build graph
const builder = new MLGraphBuilder(context);
const input = builder.input('x', { dataType: 'float32', shape: [1, 4] });
const output = builder.mul(input, builder.constant(...));
const graph = await builder.build({ 'y': output });

// Run inference (MLTensor API)
const inputTensor = await context.createTensor({ ... });
context.writeTensor(inputTensor, data);
context.dispatch(graph, inputs, outputs);
const result = await context.readTensor(outputTensor);
```

---

## 🎯 Tips for Best Results

1. **Start with WASM** to verify your model works
2. **Use float32 models** for WebNN compatibility
3. **Check input shape/type** with Netron before running
4. **Test WebNN directly** if model fails to isolate the issue
5. **Hard refresh** (Ctrl+Shift+R) after code changes

---

## 📄 License

MIT License - Feel free to use and modify for your projects.
