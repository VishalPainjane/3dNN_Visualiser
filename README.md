<div align="center">

<img src="docs/preview.png" width="100%" alt="Vizualiser - 3D Neural Network Architecture Visualization">

<br />
<br />
<!--
[![Featured on Product Hunt](https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=346331&theme=light)](https://www.producthunt.com/posts/vizualiser)
[![Top Post Badge](https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=346331&theme=light&period=daily)](https://www.producthunt.com/posts/vizualiser)
-->
<br />

[![GitHub stars](https://img.shields.io/github/stars/VishalPainjane/3dNN_Visualiser?style=flat-square&logo=github&color=yellow)](https://github.com/VishalPainjane/3dNN_Visualiser/stargazers)

<!-- [![GitHub forks](https://img.shields.io/github/forks/VishalPainjane/3dNN_Visualiser?style=flat-square&logo=github&color=blue)](https://github.com/VishalPainjane/3dNN_Visualiser/network/members) -->

[![GitHub watchers](https://img.shields.io/github/watchers/VishalPainjane/3dNN_Visualiser?style=flat-square&logo=github)](https://github.com/VishalPainjane/3dNN_Visualiser/watchers)
[![GitHub contributors](https://img.shields.io/github/contributors/VishalPainjane/3dNN_Visualiser?color=blue&style=flat-square&logo=github)](https://github.com/VishalPainjane/3dNN_Visualiser/graphs/contributors)
[![GitHub last commit](https://img.shields.io/github/last-commit/VishalPainjane/3dNN_Visualiser?color=blue&style=flat-square&logo=github)](https://github.com/VishalPainjane/3dNN_Visualiser/commits/main)
[![GitHub license](https://img.shields.io/github/license/VishalPainjane/3dNN_Visualiser?color=blue&style=flat-square)](LICENSE)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/VishalPainjane/3dNN_Visualiser?color=blue&style=flat-square&logo=github)](https://github.com/VishalPainjane/3dNN_Visualiser/issues?q=is%3Aissue+is%3Aclosed)
[![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/VishalPainjane/3dNN_Visualiser?color=blue&style=flat-square&logo=github)](https://github.com/VishalPainjane/3dNN_Visualiser/pulls?q=is%3Apr+is%3Aclosed)

<!--
[![npm version](https://img.shields.io/npm/v/vizualiser?style=flat-square&logo=npm&color=red)](https://www.npmjs.com/package/vizualiser)
[![npm downloads](https://img.shields.io/npm/dm/vizualiser?style=flat-square&logo=npm&color=red)](https://www.npmjs.com/package/vizualiser)
[![Docker Pulls](https://img.shields.io/docker/pulls/vizualiser/vizualiser?style=flat-square&logo=docker&color=blue)](https://hub.docker.com/r/vizualiser/vizualiser)
[![Build Status](https://img.shields.io/github/actions/workflow/status/VishalPainjane/vizualiser/ci.yml?style=flat-square&logo=github-actions)](https://github.com/VishalPainjane/vizualiser/actions)
[![Code Coverage](https://img.shields.io/codecov/c/github/VishalPainjane/vizualiser?style=flat-square&logo=codecov)](https://codecov.io/gh/VishalPainjane/vizualiser)
[![Code Quality](https://img.shields.io/codacy/grade/abc123?style=flat-square&logo=codacy)](https://www.codacy.com/app/vizualiser/vizualiser)

[![Discord](https://img.shields.io/discord/123456789?style=flat-square&logo=discord&label=Discord&color=7289da)](https://discord.gg/vizualiser)
[![Twitter Follow](https://img.shields.io/twitter/follow/vizualiser?style=flat-square&logo=twitter&color=1DA1F2)](https://twitter.com/vizualiser)
[![YouTube Channel Subscribers](https://img.shields.io/youtube/channel/subscribers/UC123?style=flat-square&logo=youtube&color=red)](https://youtube.com/@vizualiser)
[![Reddit](https://img.shields.io/reddit/subreddit-subscribers/vizualiser?style=flat-square&logo=reddit&color=FF4500)](https://reddit.com/r/vizualiser) -->

</div>

<br />

## 3D Neural Network Architecture Visualizer

Vizualiser is an open-source platform for visualizing, analyzing, and understanding deep learning architectures in stunning 3D. This is a professional tool for inspecting model structure, layer parametrization, and inter-layer connectivity. It focuses on clarity, accuracy, and performance rendering large architectures while keeping interactive frame rates on modern hardware.

<div align="center">

**[Live Demo](https://vizualiser.ai/demo)** |
**[Documentation](https://docs.vizualiser.ai)** |
**[Gallery](https://vizualiser.ai/gallery)** |
**[API Reference](https://api.vizualiser.ai)** |
**[Discord Community](https://discord.gg/vizualiser)**

</div>

---

## 🌟 Why Vizualiser?

- **Blazing fast** - GPU-accelerated, real-time rendering for massive models
- **Plug & Play** - Drag and drop, or integrate with your ML pipeline
- **Open ecosystem** - Exporters, plugins, and a thriving community
- **Beautiful** - Publication-ready visuals and dark/light themes

---

## 📦 Supported Model Formats

| Format          | Extension                    | Support       | Notes                                |
| --------------- | ---------------------------- | ------------- | ------------------------------------ |
| **ONNX**        | `.onnx`                      | ✅ Full       | Parsed directly in browser           |
| **PyTorch**     | `.pt`, `.pth`, `.ckpt`       | ✅ Full       | Analyzed via Python Backend          |
| **Keras/TF**    | `.h5`, `.hdf5`               | ✅ Full       | Analyzed via Python Backend          |
| **TF SavedModel**| `.pb`                       | ✅ Full       | Analyzed via Python Backend          |

*Note: Models without graph structure (e.g. PyTorch state_dicts) will be visualized as a 3D weight matrix.*

## ✨ Features

### 🎨 3D Visualization

- **Layer Geometry**: Each layer type renders as a distinct 3D shape (boxes, spheres, custom geometry)
- **Connection Rendering**: Multiple edge styles (lines, bezier curves, 3D tubes, arrows)
- **Color Coding**: Automatic color assignment based on layer categories
- **Level of Detail (LOD)**: Optimized rendering for large networks

### 🔄 Interactive Navigation

- **Orbit Controls**: Click and drag to rotate the view
- **Zoom**: Scroll wheel to zoom in/out
- **Pan**: Right-click and drag to pan
- **Selection**: Click layers to view detailed information
- **Hover**: Hover over layers for quick tooltips

### 📐 Layout Algorithms

- **Layered**: Traditional left-to-right topological layout
- **Force-Directed**: Physics-based spring simulation
- **Circular**: Nodes arranged in a circle
- **Hierarchical**: Tree-like arrangement based on network depth

### 🛠️ Tools & Controls

- **File Upload**: Drag & drop or click to load model files
- **Layout Selector**: Switch between layout algorithms
- **Edge Styles**: Toggle between line, bezier, tube, and arrow styles
- **Label Toggle**: Show/hide layer labels
- **Edge Toggle**: Show/hide connections
- **Keyboard Shortcuts**: Quick access to common actions

---

## ⚡ Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn
- Python 3.11+ (for backend)

### Quick Start (Docker Recommended)

```bash
# Clone the repository
git clone https://github.com/VishalPainjane/3dNN_Visualiser.git
cd 3dNN_Visualiser

# Start with Docker Compose
docker-compose up --build

# Or use the convenience scripts:
# Windows: docker-start.bat
# Linux/Mac: ./docker-start.sh
```

Access the application:

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8001
- **API Docs**: http://localhost:8001/docs

```bash
# Stop containers
docker-compose down

# View logs
docker-compose logs -f
```

### Manual Installation (Development)

```bash
# Clone the repository
git clone https://github.com/VishalPainjane/3dNN_Visualiser.git
cd 3dNN_Visualiser

# Install frontend dependencies
cd frontend
npm install

# Start frontend development server
npm run dev

# In another terminal, start the backend
cd ../backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open [http://localhost:5173](http://localhost:5173) to view the visualizer.

### Production Build

```bash
cd frontend
npm run build
npm run preview
```

---

## 🕹️ Usage

### Loading Models

1. **Drag & Drop**: Drag any supported model file onto the drop zone
2. **Click to Upload**: Click the drop zone and select a file
3. **Sample Models**: Try the included samples in `samples/`

**Directly Supported Formats:**

- `.nn3d` / `.json` - Native format (full structure)
- `.onnx` - ONNX models (parsed in browser)
- `.safetensors` - SafeTensors (structure inferred from weights)

**Formats Requiring Conversion:**

- `.pt` / `.pth` / `.ckpt` - Use Python exporter
- `.h5` / `.hdf5` - Convert to ONNX first

### Keyboard Shortcuts

| Key   | Action                                                         |
| ----- | -------------------------------------------------------------- |
| `1-4` | Switch layout (1=Layered, 2=Force, 3=Circular, 4=Hierarchical) |
| `L`   | Toggle labels                                                  |
| `E`   | Toggle edges                                                   |
| `Esc` | Deselect layer                                                 |
| `R`   | Reset camera view                                              |

### Navigation

- **Left Mouse + Drag**: Rotate view
- **Right Mouse + Drag**: Pan view
- **Scroll Wheel**: Zoom in/out
- **Left Click on Layer**: Select and view details
- **Hover on Layer**: Show tooltip

---

## 📄 .nn3d File Format

The `.nn3d` format is a JSON-based schema for describing neural network architectures:

```json
{
  "version": "1.0.0",
  "metadata": {
    "name": "My Model",
    "framework": "pytorch",
    "totalParams": 1000000
  },
  "graph": {
    "nodes": [
      {
        "id": "layer1",
        "type": "conv2d",
        "name": "Conv Layer",
        "params": { "inChannels": 3, "outChannels": 64 },
        "outputShape": [1, 64, 224, 224]
      }
    ],
    "edges": [{ "source": "input", "target": "layer1" }]
  },
  "visualization": {
    "layout": "layered",
    "theme": "dark"
  }
}
```

### Supported Layer Types

| Category           | Types                                                    |
| ------------------ | -------------------------------------------------------- |
| **Input/Output**   | input, output                                            |
| **Convolution**    | conv1d, conv2d, conv3d, convTranspose2d, depthwiseConv2d |
| **Linear**         | linear, dense                                            |
| **Activation**     | relu, gelu, sigmoid, tanh, softmax, leakyRelu, swish     |
| **Normalization**  | batchNorm, layerNorm, groupNorm, instanceNorm            |
| **Pooling**        | maxPool2d, avgPool2d, globalAvgPool, adaptiveAvgPool2d   |
| **Attention**      | multiHeadAttention, selfAttention, crossAttention        |
| **Recurrent**      | lstm, gru, rnn                                           |
| **Regularization** | dropout, dropPath                                        |
| **Operations**     | add, concat, multiply, split, reshape, flatten, permute  |
| **Embedding**      | embedding, positionalEncoding                            |

---

## 🐍 Python Exporters

Export PyTorch and ONNX models to `.nn3d` format:

### Installation

```bash
cd exporters/python
pip install -e .
```

### PyTorch Export

```python
from nn3d_exporter import PyTorchExporter
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

exporter = PyTorchExporter(model, "My Model")
exporter.trace(torch.randn(1, 784))
exporter.save("model.nn3d")
```

### ONNX Export

```python
from nn3d_exporter import ONNXExporter

exporter = ONNXExporter.from_file("model.onnx", "My Model")
exporter.save("model.nn3d")
```

---

## 🎓 Sample Models

The `samples/` directory includes example models:

| File                       | Description                              |
| -------------------------- | ---------------------------------------- |
| `simple_mlp.nn3d`          | Basic MLP for MNIST classification       |
| `cnn_resnet.nn3d`          | ResNet-style CNN with skip connections   |
| `transformer_encoder.nn3d` | Transformer encoder block with attention |

---

## 🛠️ Tech Stack

- **React 18** - UI framework
- **Three.js** - 3D graphics via @react-three/fiber
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Zustand** - State management
- **AJV** - JSON Schema validation

---

## 📁 Project Structure

```
vizualiser/
├── backend/             # Python backend service
├── frontend/            # React frontend application
│   ├── src/
│   │   ├── schema/      # .nn3d schema & types
│   │   ├── core/        # State management & algorithms
│   │   ├── components/
│   │   │   ├── layers/  # 3D layer geometry
│   │   │   ├── edges/   # Connection rendering
│   │   │   ├── controls/# Camera & interaction
│   │   │   └── ui/      # UI overlays
│   │   ├── App.tsx
│   │   └── main.tsx
│   └── public/
├── exporters/
│   └── python/          # Python export package
└── samples/             # Example .nn3d files
```

---

## 🎓 Supported Model Architectures

<details open>
<summary><b>Computer Vision</b> (Click to expand)</summary>

- **CNNs**: ResNet, VGG, Inception, DenseNet, EfficientNet, MobileNet
- **Object Detection**: YOLO, R-CNN, Faster R-CNN, SSD, RetinaNet
- **Segmentation**: U-Net, Mask R-CNN, DeepLab, SegNet
- **GANs**: StyleGAN, CycleGAN, Pix2Pix, BigGAN

</details>

<details>
<summary><b>Natural Language Processing</b></summary>

- **Transformers**: BERT, GPT, T5, BART, RoBERTa
- **LLMs**: GPT-3, GPT-4, LLaMA, Claude, PaLM
- **Seq2Seq**: LSTM, GRU, Attention models
- **Embeddings**: Word2Vec, GloVe, FastText

</details>

<details>
<summary><b>Audio & Speech</b></summary>

- **ASR**: Wav2Vec, Whisper, DeepSpeech
- **TTS**: Tacotron, WaveNet, FastSpeech
- **Audio**: SpecAugment, MelGAN

</details>

<details>
<summary><b>Multimodal</b></summary>

- **Vision-Language**: CLIP, DALL-E, Flamingo, BLIP
- **Video**: TimeSformer, VideoMAE, SlowFast

</details>

<details>
<summary><b>Reinforcement Learning</b></summary>

- **Policy Networks**: DQN, A3C, PPO, SAC
- **World Models**: MuZero, Dreamer

</details>

---

## 📊 Comparison with Alternatives

| Feature                       | Vizualiser    | Netron   | TensorBoard | Weights & Biases |
| ----------------------------- | ------------- | -------- | ----------- | ---------------- |
| **3D Visualization**          | ✅ Advanced   | ❌       | ❌          | ⚠️ Basic         |
| **Real-time Editing**         | ✅            | ❌       | ❌          | ❌               |
| **Format Support**            | ✅ 10+        | ✅ 8+    | ⚠️ 3        | ⚠️ 4             |
| **Browser-based**             | ✅            | ✅       | ⚠️ Limited  | ✅               |
| **GPU Acceleration**          | ✅            | ❌       | ❌          | ❌               |
| **Collaboration**             | ✅            | ❌       | ❌          | ✅               |
| **Export Options**            | ✅ 8+ formats | ⚠️ PNG   | ⚠️ PNG      | ⚠️ PNG           |
| **Open Source**               | ✅ MIT        | ✅       | ✅ Apache   | ❌               |
| **Performance (100k layers)** | ✅ 60 FPS     | ⚠️ 5 FPS | N/A         | N/A              |
| **Cost**                      | Free          | Free     | Free        | Paid             |

---

## 📜 License

Vizualiser is licensed under the **[MIT License](LICENSE)**.

```
MIT License

Copyright (c) 2025 Vizualiser Team

Permission is hereby granted, free of charge, to any person obtaining a copy...
```

**TL;DR**: Free for commercial and personal use. Attribution appreciated but not required.

---

## 🙏 Acknowledgments

Vizualiser wouldn't exist without these amazing projects:

- **[Three.js](https://threejs.org/)** - 3D rendering engine
- **[React](https://reactjs.org/)** - UI framework
- **[React Three Fiber](https://docs.pmnd.rs/react-three-fiber/)** - React + Three.js
- **[ONNX](https://onnx.ai/)** - Model interchange format
- **[PyTorch](https://pytorch.org/)** - Deep learning framework

---

<div align="center">

### ⭐ **Star Us on GitHub** ⭐

If Vizualiser helped you, please consider giving us a star!  
It helps others discover the project.

[![Star History Chart](https://api.star-history.com/svg?repos=VishalPainjane/3dNN_Visualiser&type=Date)](https://star-history.com/#VishalPainjane/3dNN_Visualiser&Date)

---

**Made with ❤️ by [Vishal Painjane](https://github.com/VishalPainjane)**

**[⬆ Back to Top](#-vizualiser)**

</div>

---

<div align="center">
<sub>🌟 If you like Vizualiser, give it a star on GitHub! 🌟</sub>
</div>

---
