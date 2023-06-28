# WebNN Test Automation

## TensorFlow.js E2E Benchmark Test for MediaPipe Models

### Config

Edit `config.json`:
- Environment infomation
- Browser infomation, e.g. installed path
- MediaPipe models and links

### How to Run

```
npm install
npm run test
```

### Log

Get log in command line:

```
Starting testing...
----------------------------------------
CPU: Intel(R) Core(TM) i7-8665U CPU @ 1.90GHz 2.11 GHz
RAM: 24.0 GB
GPU: Intel UHD Graphics 620 31.0.101.2111
OS: Windows 10 Enterprise 22H2 19045.3086
Browser: Chrome Canary, Build: 117.0.5857.0
----------------------------------------
Category: Object Detection, Model: EfficientDet-Lite0 (float 32), URL: https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/latest/efficientdet_lite0.tfliteWasm SIMD: 155.6 ms
WebNN: 54.8 ms
Category: Face Detection, Model: EfficientDet-Lite2 (float 32), URL: https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float32/latest/efficientdet_lite2.tflite  
Wasm SIMD: 489.3 ms
WebNN: 158.8 ms
Category: Face Detection, Model: SSDMobileNet-V2 (float 32), URL: https://storage.googleapis.com/mediapipe-models/object_detector/ssd_mobilenet_v2/float32/latest/ssd_mobilenet_v2.tflite
Wasm SIMD: 93.4 ms
WebNN: 32.1 ms
----------------------------------------
Testing completed
```
  
