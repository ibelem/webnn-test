/**
 * ============================================================================
 * WebNN Sample App - main.js
 * ============================================================================
 * 
 * A browser-based ONNX model inference application using ONNX Runtime Web
 * with WebNN (Web Neural Network) Execution Provider support.
 * 
 * This is similar to WinMLSampleApp but runs entirely in the browser.
 * 
 * KEY CONCEPTS:
 * -------------
 * 
 * 1. ONNX Runtime Web
 *    - JavaScript library that runs ONNX models in the browser
 *    - Loaded via CDN: onnxruntime-web@1.24.1
 *    - Provides the `ort` global object with InferenceSession, Tensor, etc.
 * 
 * 2. Execution Providers (EP)
 *    - Different backends for running model inference
 *    - WebNN: Uses browser's Web Neural Network API (NPU/GPU/CPU acceleration)
 *    - WebGPU: Uses GPU via WebGPU API
 *    - WASM: Uses WebAssembly on CPU (most compatible, slower)
 * 
 * 3. WebNN API
 *    - W3C standard for hardware-accelerated ML in browsers
 *    - Accesses NPU (Neural Processing Unit), GPU, or CPU
 *    - Still experimental - requires Chrome/Edge flag to enable
 *    - Flag: chrome://flags/#web-machine-learning-neural-network
 * 
 * 4. Model Requirements for WebNN
 *    - Float32 models work best with WebNN
 *    - Quantized models (int8/uint8) may fail due to limited operator support
 *    - Per-channel quantization is NOT supported by WebNN (use per-tensor or float32)
 * 
 * TESTED CONFIGURATION:
 * ---------------------
 * - ONNX Runtime Web: 1.24.1
 * - Chrome/Edge with WebNN flag enabled
 * - Windows 11 with NPU support
 * 
 * ============================================================================
 */

// ============================================================================
// GLOBAL STATE
// ============================================================================

/**
 * Global ONNX Runtime InferenceSession instance.
 * Created when a model is loaded, used for running inference.
 * Should be released before loading a new model to free resources.
 */
let session = null;

/**
 * Array of input tensor names from the loaded model.
 * Example: ['input'] or ['image', 'scale']
 * Retrieved from session.inputNames after model loading.
 */
let modelInputs = null;

/**
 * Array of output tensor names from the loaded model.
 * Example: ['output'] or ['boxes', 'scores']
 * Retrieved from session.outputNames after model loading.
 */
let modelOutputs = null;

// ============================================================================
// DOM ELEMENT REFERENCES
// ============================================================================

// Log output area - displays all messages to the user
const logElement = document.getElementById('log');

// Model file selection
const modelFileInput = document.getElementById('modelFile');
const loadModelBtn = document.getElementById('loadModelBtn');

// Inference controls
const runRandomBtn = document.getElementById('runRandomBtn');
const runBenchmarkBtn = document.getElementById('runBenchmarkBtn');
const clearLogBtn = document.getElementById('clearLogBtn');

// Model information display area
const modelInfoDiv = document.getElementById('modelInfo');

// Configuration selectors
const deviceTypeSelect = document.getElementById('deviceType');      // CPU/GPU/NPU
const powerPrefSelect = document.getElementById('powerPreference');  // Power mode
const epSelect = document.getElementById('executionProvider');       // WebNN/WebGPU/WASM
const iterationsInput = document.getElementById('iterations');       // Benchmark count
const inputShapeInput = document.getElementById('inputShape');       // Tensor shape
const dataTypeSelect = document.getElementById('inputDataType');     // float32/uint8/etc.

// ============================================================================
// LOGGING FUNCTIONS
// ============================================================================

/**
 * Logs a message to both the UI log area and browser console.
 * 
 * @param {string} message - The message to display
 * @param {string} type - Message type: 'info', 'success', 'error', 'warn', 'timing'
 * 
 * Each type has different color styling defined in index.html CSS:
 * - info: gray (default)
 * - success: green
 * - error: red
 * - warn: yellow
 * - timing: blue (for performance metrics)
 */
function log(message, type = 'info') {
    const timestamp = new Date().toLocaleTimeString();
    const line = document.createElement('div');
    line.className = `log-${type}`;
    line.textContent = `[${timestamp}] ${message}`;
    logElement.appendChild(line);
    
    // Auto-scroll to show latest message
    logElement.scrollTop = logElement.scrollHeight;
    
    // Also log to browser console for debugging
    console.log(`[${type.toUpperCase()}] ${message}`);
}

// Convenience wrappers for different log types
function logInfo(msg) { log(msg, 'info'); }
function logSuccess(msg) { log(msg, 'success'); }
function logError(msg) { log(msg, 'error'); }
function logWarn(msg) { log(msg, 'warn'); }
function logTiming(msg) { log(msg, 'timing'); }

// ============================================================================
// WEBNN SUPPORT CHECK
// ============================================================================

/**
 * Checks if the WebNN API is available in the current browser.
 * 
 * WebNN availability requires:
 * 1. Chrome or Edge browser (Chromium-based)
 * 2. WebNN flag enabled: chrome://flags/#web-machine-learning-neural-network
 * 3. Browser restart after enabling the flag
 * 
 * The check tries to create an MLContext which is the main entry point
 * to the WebNN API. If this succeeds, WebNN is fully available.
 * 
 * @returns {Promise<boolean>} True if WebNN is available and working
 */
async function checkWebNNSupport() {
    // Check if the 'ml' namespace exists on navigator
    // This is the entry point to WebNN: navigator.ml
    if ('ml' in navigator) {
        try {
            // Try to create an MLContext - this is the real test
            // MLContext is needed to build and run neural network graphs
            const context = await navigator.ml.createContext({ deviceType: 'cpu' });
            logSuccess('WebNN API is available');
            return true;
        } catch (e) {
            // API exists but context creation failed
            // This could happen if driver issues or unsupported hardware
            logWarn(`WebNN API found but createContext failed: ${e.message}`);
            return false;
        }
    } else {
        // navigator.ml doesn't exist - WebNN not enabled
        logWarn('WebNN API is NOT available in this browser');
        logInfo('To enable WebNN in Chrome/Edge: chrome://flags/#web-machine-learning-neural-network');
        return false;
    }
}

// ============================================================================
// DIRECT WEBNN TEST (BYPASSES ONNX RUNTIME)
// ============================================================================

/**
 * Tests WebNN directly without using ONNX Runtime.
 * 
 * This is useful for debugging to determine if:
 * - The issue is with WebNN itself (this test fails)
 * - The issue is with ONNX Runtime Web (this test passes but model fails)
 * - The issue is with the specific model (simple models work, complex don't)
 * 
 * Creates a minimal graph that computes: output = input * 2
 * Input: [1, 2, 3, 4] → Expected output: [2, 4, 6, 8]
 * 
 * WebNN API Architecture:
 * -----------------------
 * 1. MLContext: The execution context bound to a device (CPU/GPU/NPU)
 * 2. MLGraphBuilder: Used to construct the computation graph
 * 3. MLGraph: The compiled graph ready for execution
 * 4. MLTensor: Tensors for input/output data (newer API)
 * 
 * @param {string} deviceType - 'cpu', 'gpu', or 'npu'
 * @returns {Promise<boolean>} True if the test passed
 */
async function testWebNNDirect(deviceType = 'cpu') {
    logInfo(`Testing WebNN directly with device: ${deviceType}...`);
    
    try {
        // ===== STEP 1: Create MLContext =====
        // The context represents a connection to a compute device.
        // deviceType determines which hardware runs the computation:
        // - 'cpu': Software fallback, always available
        // - 'gpu': Uses GPU (DirectML on Windows, Metal on macOS)
        // - 'npu': Uses Neural Processing Unit if available
        logInfo('Creating MLContext...');
        const context = await navigator.ml.createContext({ deviceType });
        logSuccess(`MLContext created for ${deviceType}`);
        
        // ===== STEP 2: Build Computation Graph =====
        // MLGraphBuilder is a factory for creating graph operations.
        // You define inputs, constants, and operations to build a graph.
        logInfo('Building simple graph (y = x * 2)...');
        const builder = new MLGraphBuilder(context);
        
        // Define input tensor descriptor
        // - dataType: The element type (float32, float16, int32, etc.)
        // - shape: Tensor dimensions [batch, features] or [N, C, H, W] etc.
        const inputDesc = { dataType: 'float32', shape: [1, 4] };
        
        // Create input placeholder - 'x' is the name we'll use to feed data
        const input = builder.input('x', inputDesc);
        
        // Create a constant tensor with value 2.0
        // This will be multiplied with the input
        const constant = builder.constant(
            { dataType: 'float32', shape: [1] },  // Scalar broadcasted
            new Float32Array([2.0])
        );
        
        // Create multiplication operation: output = input * constant
        // builder.mul returns an MLOperand representing the result
        const output = builder.mul(input, constant);
        
        // ===== STEP 3: Compile the Graph =====
        // build() compiles the graph for the target device.
        // The object maps output names to MLOperands.
        // This step may take time for complex graphs on NPU.
        logInfo('Compiling graph...');
        const graph = await builder.build({ 'y': output });
        logSuccess('Graph compiled!');
        
        // ===== STEP 4: Run Inference =====
        // The WebNN API has evolved. Different browser versions support
        // different APIs for running inference:
        // - Newer: MLTensor with context.dispatch() and context.readTensor()
        // - Older: context.compute() or graph.compute()
        logInfo('Running inference...');
        const inputData = new Float32Array([1.0, 2.0, 3.0, 4.0]);
        const outputData = new Float32Array(4);
        
        // Try the newer MLTensor API first (Chrome 2024+)
        // MLTensor is a device-resident tensor that avoids CPU-GPU copies
        // Check that all required methods exist as functions before using this API
        const hasMLTensorAPI = typeof context.createTensor === 'function' &&
                               typeof context.writeTensor === 'function' &&
                               typeof context.dispatch === 'function' &&
                               typeof context.readTensor === 'function';
        
        if (hasMLTensorAPI) {
            logInfo('Using MLTensor API...');
            
            // Create input tensor on the device
            // writable: true means we can write data to it
            const inputTensor = await context.createTensor({
                dataType: 'float32',
                shape: [1, 4],
                writable: true
            });
            
            // Create output tensor on the device
            // readable: true means we can read results from it
            const outputTensor = await context.createTensor({
                dataType: 'float32',
                shape: [1, 4],
                readable: true
            });
            
            // Write input data to the device tensor
            await context.writeTensor(inputTensor, inputData);
            
            // Execute the graph
            // dispatch() is asynchronous and runs on the device
            await context.dispatch(graph, { 'x': inputTensor }, { 'y': outputTensor });
            
            // Read results back from device to CPU
            const result = await context.readTensor(outputTensor);
            const resultArray = new Float32Array(result);
            
            logSuccess(`WebNN ${deviceType} works! Input: [${inputData}] → Output: [${resultArray}]`);
            logInfo(`Expected: [2, 4, 6, 8], Got: [${resultArray}]`);
        } 
        // Fallback to older context.compute API
        else if (context.compute) {
            logInfo('Using context.compute API...');
            const inputs = { 'x': inputData };
            const outputs = { 'y': outputData };
            await context.compute(graph, inputs, outputs);
            
            logSuccess(`WebNN ${deviceType} works! Input: [${inputData}] → Output: [${outputData}]`);
            logInfo(`Expected: [2, 4, 6, 8], Got: [${outputData}]`);
        }
        // Try graph.compute (older API variant)
        else if (graph.compute) {
            logInfo('Using graph.compute API...');
            const inputs = { 'x': inputData };
            const outputs = { 'y': outputData };
            await graph.compute(inputs, outputs);
            
            logSuccess(`WebNN ${deviceType} works! Input: [${inputData}] → Output: [${outputData}]`);
            logInfo(`Expected: [2, 4, 6, 8], Got: [${outputData}]`);
        }
        else {
            // No compatible API found - browser version issue
            logError('No compatible WebNN compute API found');
            logInfo(`Available on context: ${Object.keys(context).join(', ')}`);
            logInfo(`Available on graph: ${Object.keys(graph).join(', ')}`);
            return false;
        }
        
        return true;
    } catch (e) {
        logError(`WebNN ${deviceType} test failed: ${e.message}`);
        console.error(e);
        return false;
    }
}

// ============================================================================
// ONNX RUNTIME SESSION OPTIONS
// ============================================================================

/**
 * Builds the session options object for ONNX Runtime Web.
 * 
 * Session options control how the model is loaded and executed.
 * The most important setting is `executionProviders` which determines
 * what hardware/backend runs the model.
 * 
 * Execution Providers Explained:
 * ------------------------------
 * 
 * 1. 'webnn' - Web Neural Network API
 *    - Best for: NPU/GPU acceleration, power efficiency
 *    - deviceType options:
 *      - 'cpu': Software implementation (fallback)
 *      - 'gpu': GPU via DirectML (Windows) or Metal (macOS)
 *      - 'npu': Neural Processing Unit (if available)
 *    - powerPreference options:
 *      - 'default': System decides
 *      - 'low-power': Prefer integrated GPU / power efficiency
 *      - 'high-performance': Prefer discrete GPU / max speed
 *    - Requirements: Chrome/Edge with WebNN flag enabled
 *    - Limitations: Not all ONNX operators supported
 * 
 * 2. 'webgpu' - WebGPU API
 *    - Best for: GPU compute, good operator coverage
 *    - Uses: GPU shaders for computation
 *    - Requirements: Chrome 113+, Edge 113+
 *    - Good balance of speed and compatibility
 * 
 * 3. 'wasm' - WebAssembly
 *    - Best for: Maximum compatibility
 *    - Uses: CPU via WebAssembly (compiled C++)
 *    - Works in: All modern browsers
 *    - Supports: All ONNX operators
 *    - Slowest option but most reliable
 * 
 * @returns {Object} Session options for ort.InferenceSession.create()
 */
function getSessionOptions() {
    // Read user selections from UI
    const ep = epSelect.value;                    // 'webnn', 'webgpu', or 'wasm'
    const deviceType = deviceTypeSelect.value;    // 'cpu', 'gpu', or 'npu'
    const powerPreference = powerPrefSelect.value; // 'default', 'low-power', 'high-performance'

    let executionProviders;
    
    if (ep === 'webnn') {
        // WebNN requires an options object with device settings
        executionProviders = [{
            name: 'webnn',
            deviceType: deviceType,           // Which hardware to use
            powerPreference: powerPreference  // Performance vs battery
        }];
    } else if (ep === 'webgpu') {
        // WebGPU is simpler - just the string name
        executionProviders = ['webgpu'];
    } else {
        // WASM fallback - always works
        executionProviders = ['wasm'];
    }

    logInfo(`Using EP: ${ep}, Device: ${deviceType}, Power: ${powerPreference}`);
    
    return {
        executionProviders: executionProviders,
        
        // Graph optimization level:
        // - 'disabled': No optimizations
        // - 'basic': Basic optimizations (constant folding, etc.)
        // - 'extended': More aggressive optimizations
        // - 'all': All available optimizations (recommended)
        graphOptimizationLevel: 'all'
    };
}

// ============================================================================
// MODEL LOADING
// ============================================================================

/**
 * Loads an ONNX model from an ArrayBuffer.
 * 
 * ONNX Model Loading Flow:
 * ------------------------
 * 1. User selects .onnx file via file input
 * 2. File is read into ArrayBuffer (binary data)
 * 3. ArrayBuffer is passed to ort.InferenceSession.create()
 * 4. ONNX Runtime parses the model, optimizes the graph
 * 5. Model is compiled for the selected execution provider
 * 6. Session is ready for inference
 * 
 * The loading time depends on:
 * - Model size (more parameters = longer load)
 * - Execution provider (NPU compilation can be slow)
 * - Graph complexity (more operators = more optimization)
 * 
 * Common Loading Errors:
 * ----------------------
 * - "Unsupported operator": EP doesn't support an op in the model
 * - "dequantizeLinear": Quantized model not compatible with WebNN
 * - Timeout/hang: Complex model taking too long to compile
 * 
 * @param {ArrayBuffer} modelData - The ONNX model file contents
 * @param {string} modelName - Display name for logging
 */
async function loadModel(modelData, modelName) {
    try {
        // Release previous session to free memory
        // This is important to avoid memory leaks
        if (session) {
            await session.release();
            session = null;
            logInfo('Previous session released');
        }

        // Get execution provider configuration
        const options = getSessionOptions();
        ort.env.logLevel = "verbose";
        
        logInfo(`Loading model: ${modelName}...`);
        const startTime = performance.now();
        
        // ===== Create the InferenceSession =====
        // This is the main ONNX Runtime API call.
        // It parses the ONNX file, builds the execution graph,
        // and compiles it for the selected EP.
        // 
        // ort.InferenceSession.create() can accept:
        // - ArrayBuffer (binary data) - what we use
        // - Uint8Array (byte array)
        // - URL string (fetches the model)
        session = await ort.InferenceSession.create(modelData, options);
        
        const loadTime = performance.now() - startTime;
        logSuccess(`Model loaded in ${loadTime.toFixed(2)} ms`);

        // ===== Get Model Metadata =====
        // After loading, we can inspect the model's inputs and outputs.
        // inputNames/outputNames are arrays of strings.
        // This tells us what tensors the model expects/produces.
        modelInputs = session.inputNames;
        modelOutputs = session.outputNames;

        // Build DOM elements to display model information safely
        // Using textContent instead of innerHTML to prevent HTML injection
        // from maliciously crafted model input/output names
        modelInfoDiv.innerHTML = '';  // Clear previous content
        
        const header = document.createElement('h4');
        header.textContent = 'Model Information:';
        modelInfoDiv.appendChild(header);
        
        // Display inputs
        const inputsLabel = document.createElement('p');
        const inputsStrong = document.createElement('strong');
        inputsStrong.textContent = `Inputs (${modelInputs.length}):`;
        inputsLabel.appendChild(inputsStrong);
        modelInfoDiv.appendChild(inputsLabel);
        
        const inputsList = document.createElement('ul');
        for (const name of modelInputs) {
            const li = document.createElement('li');
            li.textContent = name;  // Safe: textContent escapes HTML
            inputsList.appendChild(li);
        }
        modelInfoDiv.appendChild(inputsList);
        
        // Display outputs
        const outputsLabel = document.createElement('p');
        const outputsStrong = document.createElement('strong');
        outputsStrong.textContent = `Outputs (${modelOutputs.length}):`;
        outputsLabel.appendChild(outputsStrong);
        modelInfoDiv.appendChild(outputsLabel);
        
        const outputsList = document.createElement('ul');
        for (const name of modelOutputs) {
            const li = document.createElement('li');
            li.textContent = name;  // Safe: textContent escapes HTML
            outputsList.appendChild(li);
        }
        modelInfoDiv.appendChild(outputsList);

        // Enable inference buttons now that model is ready
        runRandomBtn.disabled = false;
        runBenchmarkBtn.disabled = false;

        logSuccess('Model ready for inference');
        
    } catch (error) {
        // Log the error and display it in the UI
        // Using textContent to prevent HTML injection from error messages
        logError(`Failed to load model: ${error.message}`);
        console.error(error);
        modelInfoDiv.innerHTML = '';  // Clear previous content
        const errorP = document.createElement('p');
        errorP.style.color = '#f14c4c';
        errorP.textContent = `Error: ${error.message}`;
        modelInfoDiv.appendChild(errorP);
    }
}

// ============================================================================
// TENSOR CREATION
// ============================================================================

/**
 * Creates a random tensor with the specified shape and data type.
 * 
 * Tensors in ONNX Runtime Web:
 * ----------------------------
 * - Created using: new ort.Tensor(type, data, shape)
 * - type: Data type string ('float32', 'int32', 'uint8', etc.)
 * - data: TypedArray with the actual values
 * - shape: Array of dimensions [N, C, H, W] etc.
 * 
 * Common Tensor Shapes:
 * ---------------------
 * - [1, 3, 224, 224]: Batch=1, Channels=3, Height=224, Width=224 (NCHW)
 * - [1, 224, 224, 3]: Batch=1, Height=224, Width=224, Channels=3 (NHWC)
 * - [1, 256, 288, 3]: Common for streaming models
 * 
 * Data Types:
 * -----------
 * - float32: 32-bit floating point (most common for neural networks)
 * - float16: 16-bit floating point (smaller, faster, less precision)
 * - int32: 32-bit integer
 * - int8/uint8: 8-bit integer (used in quantized models)
 * 
 * @param {number[]} shape - Tensor dimensions, e.g., [1, 3, 224, 224]
 * @param {string} dataType - Element type: 'float32', 'uint8', etc.
 * @returns {ort.Tensor} ONNX Runtime Tensor object
 */
function createRandomTensor(shape, dataType = 'float32') {
    // Calculate total number of elements
    // e.g., [1, 3, 224, 224] = 1 * 3 * 224 * 224 = 150528 elements
    const size = shape.reduce((a, b) => a * b, 1);
    
    let data;
    
    // Create appropriate TypedArray based on data type
    switch (dataType) {
        case 'float32':
            // Float32Array for 32-bit floating point
            // Random values between 0 and 1
            data = new Float32Array(size);
            for (let i = 0; i < size; i++) {
                data[i] = Math.random();
            }
            break;
            
        case 'int32':
            // Int32Array for 32-bit signed integers
            // Random values 0-255 (typical for image-like data)
            data = new Int32Array(size);
            for (let i = 0; i < size; i++) {
                data[i] = Math.floor(Math.random() * 256);
            }
            break;
            
        case 'int64':
            // BigInt64Array for 64-bit integers
            // Note: JavaScript requires BigInt for 64-bit integers
            data = new BigInt64Array(size);
            for (let i = 0; i < size; i++) {
                data[i] = BigInt(Math.floor(Math.random() * 256));
            }
            break;
            
        case 'uint8':
            // Uint8Array for 8-bit unsigned integers
            // Common for quantized models and image input (0-255)
            data = new Uint8Array(size);
            for (let i = 0; i < size; i++) {
                data[i] = Math.floor(Math.random() * 256);
            }
            break;
            
        case 'int8':
            // Int8Array for 8-bit signed integers (-128 to 127)
            data = new Int8Array(size);
            for (let i = 0; i < size; i++) {
                data[i] = Math.floor(Math.random() * 256) - 128;
            }
            break;
            
        default:
            // Default to float32 for unknown types
            data = new Float32Array(size);
            for (let i = 0; i < size; i++) {
                data[i] = Math.random();
            }
    }
    
    // Create and return the ONNX Runtime Tensor
    // The tensor wraps the TypedArray with shape information
    return new ort.Tensor(dataType, data, shape);
}

// ============================================================================
// INFERENCE
// ============================================================================

/**
 * Runs inference on the loaded model with random input data.
 * 
 * Inference Flow:
 * ---------------
 * 1. Parse input shape from UI (e.g., "1,3,400,400" → [1, 3, 400, 400])
 * 2. Create random input tensor(s) with specified shape and type
 * 3. Build feeds object: { inputName: tensor }
 * 4. Call session.run(feeds) to execute the model
 * 5. Process and display output tensor information
 * 
 * The session.run() method:
 * -------------------------
 * - Takes a "feeds" object mapping input names to tensors
 * - Returns a "results" object mapping output names to tensors
 * - Runs synchronously (blocks until complete)
 * - Automatically uses the EP configured at session creation
 * 
 * @returns {Object|null} Object with {results, inferenceTime} or null on error
 */
async function runInference() {
    // Check if a model has been loaded
    if (!session) {
        logError('No model loaded');
        return null;
    }

    try {
        // The "feeds" object maps input names to input tensors
        // For a model with input named "input", it looks like:
        // { "input": ort.Tensor }
        const feeds = {};
        
        // ===== Parse Input Shape from UI =====
        // User enters shape as comma-separated values: "1,3,400,400"
        // We parse this into an array of integers: [1, 3, 400, 400]
        const shapeStr = inputShapeInput.value.trim();
        const inputShape = shapeStr.split(',').map(s => parseInt(s.trim()));
        
        // Validate the parsed shape
        if (inputShape.some(isNaN) || inputShape.length === 0) {
            logError(`Invalid input shape: ${shapeStr}`);
            return null;
        }
        
        // Get the selected data type from UI dropdown
        const dataType = dataTypeSelect.value;
        
        // ===== Create Input Tensors =====
        // Most models have one input, but some have multiple.
        // We create a tensor for each input with the specified shape.
        // 
        // Note: In production, you would use real data instead of random.
        // This demo uses random data just to test model execution.
        for (const inputName of modelInputs) {
            logInfo(`Creating random input for "${inputName}" with shape [${inputShape}], type=${dataType}`);
            feeds[inputName] = createRandomTensor(inputShape, dataType);
        }

        // ===== Run Model Inference =====
        logInfo('Running inference...');
        const startTime = performance.now();
        
        // session.run() executes the model
        // - Input: feeds object with named tensors
        // - Output: results object with named output tensors
        // This is where the actual computation happens on the selected EP
        const results = await session.run(feeds);
        
        const inferenceTime = performance.now() - startTime;
        logTiming(`Inference completed in ${inferenceTime.toFixed(2)} ms`);

        // ===== Process and Display Results =====
        // Iterate through each output tensor and log its info
        for (const [name, tensor] of Object.entries(results)) {
            // tensor.dims: Array of dimensions, e.g., [1, 3, 400, 400]
            // tensor.type: Data type string, e.g., 'float32'
            logInfo(`Output "${name}": shape=[${tensor.dims}], type=${tensor.type}`);
            
            // Show a preview of the first 5 values
            // tensor.data is the underlying TypedArray
            const data = tensor.data;
            const preview = Array.from(data.slice(0, 5)).map(v => 
                typeof v === 'number' ? v.toFixed(4) : v.toString()
            );
            logInfo(`  First 5 values: [${preview.join(', ')}...]`);
        }

        return { results, inferenceTime };
        
    } catch (error) {
        logError(`Inference failed: ${error.message}`);
        console.error(error);
        return null;
    }
}

// ============================================================================
// BENCHMARKING
// ============================================================================

/**
 * Runs multiple inference iterations to measure performance.
 * 
 * Benchmarking Methodology:
 * -------------------------
 * 1. Warmup run: First run is often slower (JIT compilation, caching)
 * 2. Timed runs: Multiple iterations to get statistical significance
 * 3. Calculate: Average, min, max, and FPS
 * 
 * Why benchmark multiple times?
 * - First run often includes one-time setup costs
 * - Performance can vary due to system load
 * - Average gives more reliable estimate than single run
 */
async function runBenchmark() {
    if (!session) {
        logError('No model loaded');
        return;
    }

    // Get number of iterations from UI
    const iterations = parseInt(iterationsInput.value) || 10;
    logInfo(`Starting benchmark: ${iterations} iterations`);

    const times = [];  // Array to collect timing data
    
    // ===== Warmup Run =====
    // The first inference often includes:
    // - Shader compilation (WebGPU)
    // - NPU initialization (WebNN)
    // - Memory allocation
    // We exclude this from measurements
    logInfo('Warmup run...');
    await runInference();

    // ===== Benchmark Runs =====
    for (let i = 0; i < iterations; i++) {
        const result = await runInference();
        if (result) {
            times.push(result.inferenceTime);
        }
    }

    // ===== Calculate Statistics =====
    if (times.length > 0) {
        const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
        const minTime = Math.min(...times);
        const maxTime = Math.max(...times);
        
        // FPS = 1000ms / avgTime(ms)
        const fps = 1000 / avgTime;

        logSuccess('=== Benchmark Results ===');
        logTiming(`Iterations: ${times.length}`);
        logTiming(`Average: ${avgTime.toFixed(2)} ms`);
        logTiming(`Min: ${minTime.toFixed(2)} ms`);
        logTiming(`Max: ${maxTime.toFixed(2)} ms`);
        logTiming(`FPS: ${fps.toFixed(2)}`);
    }
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================

// Enable Load button when a file is selected
modelFileInput.addEventListener('change', (e) => {
    loadModelBtn.disabled = !e.target.files.length;
});

// Load model when button clicked
loadModelBtn.addEventListener('click', async () => {
    const file = modelFileInput.files[0];
    if (!file) return;
    
    // Read the file as ArrayBuffer (binary data)
    logInfo(`Reading file: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`);
    const arrayBuffer = await file.arrayBuffer();
    
    // Load the model
    await loadModel(arrayBuffer, file.name);
});

// Run inference button
runRandomBtn.addEventListener('click', runInference);

// Run benchmark button
runBenchmarkBtn.addEventListener('click', runBenchmark);

// Clear log button
clearLogBtn.addEventListener('click', () => {
    logElement.innerHTML = '';
});

// Test WebNN directly button
const testWebNNBtn = document.getElementById('testWebNNBtn');
testWebNNBtn.addEventListener('click', async () => {
    const deviceType = deviceTypeSelect.value;
    await testWebNNDirect(deviceType);
});

// ============================================================================
// INITIALIZATION
// ============================================================================

/**
 * Initializes the application on page load.
 * 
 * Checks:
 * - ONNX Runtime Web version
 * - WebNN API availability
 * - Browser information
 */
async function init() {
    logInfo('WebNN Sample App initialized');
    
    // Log ONNX Runtime Web version
    // ort.env.versions contains version info
    logInfo(`ONNX Runtime Web version: ${ort.env.versions?.web || 'unknown'}`);
    
    // Check if WebNN is available
    await checkWebNNSupport();
    
    // Log browser info for debugging
    logInfo(`User Agent: ${navigator.userAgent.substring(0, 80)}...`);
}

// Start the app
init();
