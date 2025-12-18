# NN3D Visualizer Backend

Python microservice for analyzing neural network model architectures.

## Features

- **PyTorch Model Analysis**: Extracts architecture information from `.pt`, `.pth`, `.ckpt`, `.bin` files
- **ONNX Model Analysis**: Parses ONNX model graphs
- **Shape Inference**: Traces model execution to capture input/output shapes
- **Layer Type Detection**: Identifies layer types from weight names and shapes

## Requirements

- Python 3.9+
- PyTorch 2.0+

## Quick Start

### Windows

```batch
cd backend
start.bat
```

### Linux/Mac

```bash
cd backend
chmod +x start.sh
./start.sh
```

### Manual Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check

```
GET /health
```

Returns server status and PyTorch version.

### Analyze PyTorch Model

```
POST /analyze
Content-Type: multipart/form-data

file: <model_file>
input_shape: (optional) comma-separated integers, e.g., "1,3,224,224"
```

### Analyze ONNX Model

```
POST /analyze/onnx
Content-Type: multipart/form-data

file: <onnx_file>
```

## Response Format

```json
{
  "success": true,
  "model_type": "full_model|state_dict|torchscript|checkpoint",
  "architecture": {
    "name": "model_name",
    "framework": "pytorch",
    "totalParameters": 1000000,
    "trainableParameters": 1000000,
    "inputShape": [1, 3, 224, 224],
    "outputShape": [1, 1000],
    "layers": [
      {
        "id": "layer_0",
        "name": "conv1",
        "type": "Conv2d",
        "category": "convolution",
        "inputShape": [1, 3, 224, 224],
        "outputShape": [1, 64, 112, 112],
        "params": {
          "in_channels": 3,
          "out_channels": 64,
          "kernel_size": [7, 7],
          "stride": [2, 2],
          "padding": [3, 3]
        },
        "numParameters": 9408,
        "trainable": true
      }
    ],
    "connections": [
      {
        "source": "layer_0",
        "target": "layer_1",
        "tensorShape": [1, 64, 112, 112]
      }
    ]
  },
  "message": "Successfully analyzed model"
}
```

## Integration with Frontend

The frontend automatically detects if the backend is available:

1. Start the backend server (port 8000)
2. Start the frontend dev server (port 3000)
3. Drop a PyTorch model file - it will use the backend for analysis

If the backend is unavailable, the frontend falls back to JavaScript-based parsing.

## Supported Layer Types

### Convolution

- Conv1d, Conv2d, Conv3d
- ConvTranspose1d, ConvTranspose2d, ConvTranspose3d

### Pooling

- MaxPool1d/2d/3d, AvgPool1d/2d/3d
- AdaptiveAvgPool1d/2d/3d, AdaptiveMaxPool1d/2d/3d

### Linear

- Linear, LazyLinear, Bilinear

### Normalization

- BatchNorm1d/2d/3d
- LayerNorm, GroupNorm, InstanceNorm

### Activation

- ReLU, LeakyReLU, PReLU, ELU, SELU
- GELU, Sigmoid, Tanh, Softmax, SiLU, Mish

### Recurrent

- RNN, LSTM, GRU

### Attention

- MultiheadAttention, Transformer, TransformerEncoder/Decoder

### Embedding

- Embedding, EmbeddingBag

### Regularization

- Dropout, Dropout2d/3d, AlphaDropout

## Architecture

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   └── model_analyzer.py  # PyTorch model analysis
├── requirements.txt
├── start.bat             # Windows startup script
├── start.sh              # Linux/Mac startup script
└── README.md
```

## Development

API documentation is available at `http://localhost:8000/docs` when the server is running.
