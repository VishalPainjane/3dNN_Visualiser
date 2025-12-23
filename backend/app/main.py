"""
NN3D Visualizer Backend API
FastAPI server for analyzing neural network models.
"""

import os
import hashlib
import tempfile
import traceback
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .model_analyzer import (
    load_pytorch_model,
    analyze_model_structure,
    analyze_state_dict,
    trace_model_shapes,
    architecture_to_dict
)

from . import database as db

import torch


app = FastAPI(
    title="NN3D Model Analyzer",
    description="Backend service for analyzing neural network model architectures",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisRequest(BaseModel):
    """Request model for analysis with sample input shape."""
    input_shape: Optional[List[int]] = None


class AnalysisResponse(BaseModel):
    """Response model for analysis results."""
    success: bool
    model_type: str
    architecture: dict
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    pytorch_version: str
    cuda_available: bool


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check server health and PyTorch availability."""
    return HealthResponse(
        status="healthy",
        pytorch_version=torch.__version__,
        cuda_available=torch.cuda.is_available()
    )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_model(
    file: UploadFile = File(...),
    input_shape: Optional[str] = Query(None, description="Input shape as comma-separated ints, e.g., '1,3,224,224'")
):
    """
    Analyze a PyTorch model file and extract architecture information.
    
    Supports:
    - Full model files (.pt, .pth)
    - State dict checkpoints
    - TorchScript models
    - Training checkpoints with model_state_dict
    """
    # Validate file extension
    allowed_extensions = {'.pt', '.pth', '.ckpt', '.bin', '.model'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Parse input shape if provided
    sample_shape = None
    if input_shape:
        try:
            sample_shape = [int(x.strip()) for x in input_shape.split(',')]
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid input_shape format. Use comma-separated integers, e.g., '1,3,224,224'"
            )
    
    # Save uploaded file temporarily
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Load and analyze model
        model, state_dict, model_type = load_pytorch_model(temp_path)
        
        if model is not None:
            # Full model available - analyze structure
            model_name = Path(file.filename).stem
            architecture = analyze_model_structure(model, model_name)
            
            # Try to trace shapes if input shape provided
            if sample_shape and model_type != 'torchscript':
                try:
                    sample_input = torch.randn(*sample_shape)
                    architecture = trace_model_shapes(model, sample_input, architecture)
                except Exception as e:
                    print(f"Shape tracing failed: {e}")
            
            return AnalysisResponse(
                success=True,
                model_type=model_type,
                architecture=architecture_to_dict(architecture),
                message=f"Successfully analyzed {model_type} model"
            )
        
        elif state_dict is not None:
            # Only state dict available - infer from weights
            model_name = Path(file.filename).stem
            architecture = analyze_state_dict(state_dict, model_name)
            
            return AnalysisResponse(
                success=True,
                model_type='state_dict',
                architecture=architecture_to_dict(architecture),
                message="Analyzed from state dict. Layer types inferred from weight names/shapes."
            )
        
        else:
            raise HTTPException(
                status_code=400,
                detail="Could not parse model file. Unknown format."
            )
    
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )
    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass


@app.post("/analyze/onnx")
async def analyze_onnx_model(file: UploadFile = File(...)):
    """
    Analyze an ONNX model file.
    """
    if not file.filename.lower().endswith('.onnx'):
        raise HTTPException(status_code=400, detail="File must be an ONNX model (.onnx)")
    
    temp_path = None
    try:
        import onnx
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.onnx') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        model = onnx.load(temp_path)
        graph = model.graph
        
        layers = []
        connections = []
        layer_map = {}
        
        # Process nodes
        for i, node in enumerate(graph.node):
            layer_id = f"layer_{i}"
            layer_map[node.name if node.name else f"node_{i}"] = layer_id
            
            # Map output names to layer ids
            for output in node.output:
                layer_map[output] = layer_id
            
            # Extract attributes
            params = {}
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.INT:
                    params[attr.name] = attr.i
                elif attr.type == onnx.AttributeProto.INTS:
                    params[attr.name] = list(attr.ints)
                elif attr.type == onnx.AttributeProto.FLOAT:
                    params[attr.name] = attr.f
                elif attr.type == onnx.AttributeProto.STRING:
                    params[attr.name] = attr.s.decode('utf-8')
            
            layers.append({
                'id': layer_id,
                'name': node.name if node.name else node.op_type,
                'type': node.op_type,
                'category': infer_onnx_category(node.op_type),
                'inputShape': None,
                'outputShape': None,
                'params': params,
                'numParameters': 0,
                'trainable': True
            })
            
            # Create connections from inputs
            for input_name in node.input:
                if input_name in layer_map:
                    source_id = layer_map[input_name]
                    if source_id != layer_id:  # Avoid self-loops
                        connections.append({
                            'source': source_id,
                            'target': layer_id,
                            'tensorShape': None
                        })
        
        # Get input/output shapes from graph
        input_shape = None
        output_shape = None
        
        if graph.input:
            for inp in graph.input:
                shape = []
                if inp.type.tensor_type.shape.dim:
                    for dim in inp.type.tensor_type.shape.dim:
                        shape.append(dim.dim_value if dim.dim_value else -1)
                if shape:
                    input_shape = shape
                    break
        
        if graph.output:
            for out in graph.output:
                shape = []
                if out.type.tensor_type.shape.dim:
                    for dim in out.type.tensor_type.shape.dim:
                        shape.append(dim.dim_value if dim.dim_value else -1)
                if shape:
                    output_shape = shape
                    break
        
        architecture = {
            'name': Path(file.filename).stem,
            'framework': 'onnx',
            'totalParameters': 0,
            'trainableParameters': 0,
            'inputShape': input_shape,
            'outputShape': output_shape,
            'layers': layers,
            'connections': connections
        }
        
        return AnalysisResponse(
            success=True,
            model_type='onnx',
            architecture=architecture,
            message="Successfully analyzed ONNX model"
        )
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"ONNX analysis failed: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass


def infer_onnx_category(op_type: str) -> str:
    """Infer category from ONNX operator type."""
    op_lower = op_type.lower()
    
    if 'conv' in op_lower:
        return 'convolution'
    if 'pool' in op_lower:
        return 'pooling'
    if 'norm' in op_lower or 'batch' in op_lower:
        return 'normalization'
    if 'relu' in op_lower or 'sigmoid' in op_lower or 'tanh' in op_lower or 'softmax' in op_lower:
        return 'activation'
    if 'gemm' in op_lower or 'matmul' in op_lower or 'linear' in op_lower:
        return 'linear'
    if 'lstm' in op_lower or 'gru' in op_lower or 'rnn' in op_lower:
        return 'recurrent'
    if 'attention' in op_lower:
        return 'attention'
    if 'dropout' in op_lower:
        return 'regularization'
    if 'reshape' in op_lower or 'flatten' in op_lower or 'squeeze' in op_lower:
        return 'reshape'
    if 'add' in op_lower or 'mul' in op_lower or 'sub' in op_lower:
        return 'arithmetic'
    if 'concat' in op_lower or 'split' in op_lower:
        return 'merge'
    
    return 'other'


# Mapping of file extensions to supported frameworks
SUPPORTED_FORMATS = {
    # Tier 1: Platinum Path (Native/Client-side preferred)
    '.nn3d': 'nn3d',
    '.json': 'nn3d',
    '.onnx': 'onnx',
    
    # Tier 2: Gold Path (Structure Inference)
    '.safetensors': 'safetensors',
    '.h5': 'keras',
    '.hdf5': 'keras',
    '.keras': 'keras',
    
    # Tier 3 & 4: Silver/Bronze Path (Backend Tracing/Stack)
    '.pt': 'pytorch',
    '.pth': 'pytorch',
    '.ckpt': 'pytorch',
    '.bin': 'pytorch',
    '.model': 'pytorch',
    
    # TensorFlow (Legacy)
    '.pb': 'tensorflow',
}


@app.post("/upload", response_model=AnalysisResponse)
async def upload_model(
    file: UploadFile = File(...),
    input_shape: Optional[str] = Query(None, description="Input shape as comma-separated ints, e.g., '1,3,224,224'")
):
    """
    Universal /upload endpoint implementing the 4-Tier Pipeline Switch.
    
    Tiers:
    1. Platinum Path (.nn3d, .onnx): Native/Zero-Latency.
       - Ideally processed client-side. If received here, it's a fallback.
    2. Gold Path (.safetensors, .h5): Structure Inference.
       - Hierarchical parsing from parameter names/config.
    3. Silver Path (.pt - JIT): Backend Tracing.
       - Full graph extraction via TorchScript.
    4. Bronze Path (.pt - State Dict): Stack Visualization.
       - Weights-only, no connectivity data.
    """
    filename = file.filename or "unknown"
    file_ext = Path(filename).suffix.lower()
    
    # Parse input shape if provided
    sample_shape = None
    if input_shape:
        try:
            sample_shape = [int(x.strip()) for x in input_shape.split(',')]
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid input_shape format. Use comma-separated integers, e.g., '1,3,224,224'"
            )

    # The "Switch" (Detect & Dispatch)
    temp_path = None
    try:
        print(f"INFO: Received file: {filename} ({file_ext})", flush=True)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        print(f"INFO: Saved temp file to {temp_path}", flush=True)
        
        result = None
        
        # Platinum Path (.nn3d, .onnx)
        if file_ext in ['.nn3d', '.json']:
            print("INFO: Dispatching to NN3D analyzer", flush=True)
            result = await _analyze_nn3d(temp_path, filename)
        
        elif file_ext == '.onnx':
            print("INFO: Dispatching to ONNX analyzer", flush=True)
            result = await _analyze_onnx(temp_path, filename)

        # Gold Path (.safetensors, .h5)
        elif file_ext in ['.safetensors']:
            print("INFO: Dispatching to SafeTensors analyzer", flush=True)
            result = await _analyze_safetensors(temp_path, filename)
            
        elif file_ext in ['.h5', '.hdf5', '.keras']:
            print("INFO: Dispatching to Keras analyzer", flush=True)
            result = await _analyze_keras(temp_path, filename)

        # Silver & Bronze Paths (.pt, .pth, etc.)
        elif file_ext in ['.pt', '.pth', '.ckpt', '.bin', '.model']:
            print("INFO: Dispatching to PyTorch analyzer (Silver/Bronze path)", flush=True)
            result = await _analyze_pytorch(temp_path, filename, sample_shape)
            
        # TensorFlow Legacy
        elif file_ext == '.pb':
            print("INFO: Dispatching to TensorFlow analyzer", flush=True)
            result = await _analyze_tensorflow(temp_path, filename)

        else:
            supported = ', '.join(sorted(SUPPORTED_FORMATS.keys()))
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format '{file_ext}'. Supported formats: {supported}"
            )
            
        if result:
            layer_count = len(result.architecture.get('layers', []))
            conn_count = len(result.architecture.get('connections', []))
            print(f"INFO: Analysis complete. Type={result.model_type}, Layers={layer_count}, Connections={conn_count}", flush=True)
            if layer_count == 0:
                print("WARNING: Model analyzed but 0 layers found!", flush=True)
            
        return result

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass


@app.post("/analyze/universal")
async def analyze_universal(
    file: UploadFile = File(...),
    input_shape: Optional[str] = Query(None, description="Input shape as comma-separated ints, e.g., '1,3,224,224'")
):
    """
    Legacy alias for /upload.
    """
    return await upload_model(file, input_shape)


async def _analyze_nn3d(file_path: str, filename: str) -> AnalysisResponse:
    """Analyze NN3D/JSON native format."""
    import json
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Basic validation: Check if it looks like NN3D
        if 'graph' not in data or 'nodes' not in data.get('graph', {}):
            raise ValueError("Invalid NN3D format: Missing 'graph.nodes'")
            
        # Extract metadata
        metadata = data.get('metadata', {})
        architecture = {
            'name': metadata.get('name', Path(filename).stem),
            'framework': metadata.get('framework', 'nn3d'),
            'totalParameters': metadata.get('totalParams', 0),
            'trainableParameters': metadata.get('trainableParams', 0),
            'inputShape': metadata.get('inputShape'),
            'outputShape': metadata.get('outputShape'),
            'layers': [], # We don't need to convert back to "layers" for NN3D files, 
                          # but AnalysisResponse expects 'architecture'. 
                          # Ideally, the frontend just uses the file directly.
            'connections': []
        }
        
        # If we need to populate layers for consistency (though frontend might ignore it if it just wants the file back)
        # But actually, if the user uploaded NN3D, they probably want it validated or re-served.
        # For now, we'll return a success message and the raw architecture if possible, 
        # or just the metadata.
        
        return AnalysisResponse(
            success=True,
            model_type='nn3d',
            architecture=architecture, # Returning minimal arch as we don't need to reverse-engineer NN3D
            message="Valid NN3D model (Platinum Path)"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid NN3D file: {str(e)}")



async def _analyze_pytorch(file_path: str, filename: str, sample_shape: Optional[List[int]] = None) -> AnalysisResponse:
    """Analyze PyTorch model."""
    model, state_dict, model_type, warning = load_pytorch_model(file_path)
    model_name = Path(filename).stem
    
    msg_suffix = f" [WARNING: {warning}]" if warning else ""
    
    if model is not None:
        architecture = analyze_model_structure(model, model_name)
        
        # Try to trace shapes if input shape provided
        if sample_shape and model_type != 'torchscript':
            try:
                sample_input = torch.randn(*sample_shape)
                architecture = trace_model_shapes(model, sample_input, architecture)
            except Exception as e:
                print(f"Shape tracing failed: {e}")
        
        return AnalysisResponse(
            success=True,
            model_type=model_type,
            architecture=architecture_to_dict(architecture),
            message=f"Successfully analyzed PyTorch {model_type}{msg_suffix}"
        )
    
    elif state_dict is not None:
        architecture = analyze_state_dict(state_dict, model_name)
        return AnalysisResponse(
            success=True,
            model_type='state_dict',
            architecture=architecture_to_dict(architecture),
            message=f"Analyzed from state dict. Layer types inferred from weight names/shapes.{msg_suffix}"
        )
    
    raise HTTPException(status_code=400, detail="Could not parse PyTorch model file")


async def _analyze_onnx(file_path: str, filename: str) -> AnalysisResponse:
    """Analyze ONNX model."""
    try:
        import onnx
    except ImportError:
        raise HTTPException(status_code=500, detail="ONNX library not installed. Install with: pip install onnx")
    
    model = onnx.load(file_path)
    graph = model.graph
    
    layers = []
    connections = []
    layer_map = {}
    total_params = 0
    
    # Process initializers (weights) for parameter counts
    weight_shapes = {}
    for init in graph.initializer:
        dims = list(init.dims)
        weight_shapes[init.name] = dims
        total_params += int(torch.prod(torch.tensor(dims)).item()) if dims else 0
    
    # Process nodes
    for i, node in enumerate(graph.node):
        layer_id = f"layer_{i}"
        layer_map[node.name if node.name else f"node_{i}"] = layer_id
        
        for output in node.output:
            layer_map[output] = layer_id
        
        # Extract attributes
        params = {}
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.INT:
                params[attr.name] = attr.i
            elif attr.type == onnx.AttributeProto.INTS:
                params[attr.name] = list(attr.ints)
            elif attr.type == onnx.AttributeProto.FLOAT:
                params[attr.name] = round(attr.f, 6)
            elif attr.type == onnx.AttributeProto.STRING:
                params[attr.name] = attr.s.decode('utf-8')
        
        # Count parameters for this layer
        layer_params = 0
        input_shapes = []
        for inp_name in node.input:
            if inp_name in weight_shapes:
                layer_params += int(torch.prod(torch.tensor(weight_shapes[inp_name])).item())
                input_shapes.append(weight_shapes[inp_name])
        
        # Infer input/output shapes from value_info
        input_shape = None
        output_shape = None
        
        layers.append({
            'id': layer_id,
            'name': node.name if node.name else f"{node.op_type}_{i}",
            'type': node.op_type,
            'category': infer_onnx_category(node.op_type),
            'inputShape': input_shape,
            'outputShape': output_shape,
            'params': params,
            'numParameters': layer_params,
            'trainable': layer_params > 0
        })
        
        # Create connections
        for input_name in node.input:
            if input_name in layer_map:
                source_id = layer_map[input_name]
                if source_id != layer_id:
                    connections.append({
                        'source': source_id,
                        'target': layer_id,
                        'tensorShape': weight_shapes.get(input_name)
                    })
    
    # Get model input/output shapes
    input_shape = None
    output_shape = None
    
    if graph.input:
        for inp in graph.input:
            if inp.name not in weight_shapes:  # Skip weight inputs
                shape = []
                if inp.type.tensor_type.shape.dim:
                    for dim in inp.type.tensor_type.shape.dim:
                        shape.append(dim.dim_value if dim.dim_value else -1)
                if shape:
                    input_shape = shape
                    break
    
    if graph.output:
        for out in graph.output:
            shape = []
            if out.type.tensor_type.shape.dim:
                for dim in out.type.tensor_type.shape.dim:
                    shape.append(dim.dim_value if dim.dim_value else -1)
            if shape:
                output_shape = shape
                break
    
    architecture = {
        'name': Path(filename).stem,
        'framework': 'onnx',
        'totalParameters': total_params,
        'trainableParameters': total_params,
        'inputShape': input_shape,
        'outputShape': output_shape,
        'layers': layers,
        'connections': connections
    }
    
    return AnalysisResponse(
        success=True,
        model_type='onnx',
        architecture=architecture,
        message=f"Successfully analyzed ONNX model with {len(layers)} layers"
    )


async def _analyze_keras(file_path: str, filename: str) -> AnalysisResponse:
    """Analyze Keras/HDF5 model."""
    try:
        import h5py
    except ImportError:
        raise HTTPException(status_code=500, detail="h5py not installed. Install with: pip install h5py")
    
    layers = []
    connections = []
    total_params = 0
    
    with h5py.File(file_path, 'r') as f:
        # Check for Keras model structure
        if 'model_config' in f.attrs:
            import json
            config = json.loads(f.attrs['model_config'])
            model_name = config.get('config', {}).get('name', Path(filename).stem)
            
            # Parse layers from config
            layer_configs = config.get('config', {}).get('layers', [])
            
            for i, layer_cfg in enumerate(layer_configs):
                layer_id = f"layer_{i}"
                layer_class = layer_cfg.get('class_name', 'Unknown')
                layer_config = layer_cfg.get('config', {})
                
                # Extract parameters
                params = {}
                param_keys = ['units', 'filters', 'kernel_size', 'strides', 'padding', 
                             'activation', 'use_bias', 'dropout', 'rate', 'axis',
                             'epsilon', 'momentum', 'input_dim', 'output_dim']
                for key in param_keys:
                    if key in layer_config:
                        params[key] = layer_config[key]
                
                # Infer shapes from config
                input_shape = None
                output_shape = None
                if 'batch_input_shape' in layer_config:
                    input_shape = list(layer_config['batch_input_shape'])
                
                layers.append({
                    'id': layer_id,
                    'name': layer_config.get('name', f"{layer_class}_{i}"),
                    'type': layer_class,
                    'category': _infer_keras_category(layer_class),
                    'inputShape': input_shape,
                    'outputShape': output_shape,
                    'params': params,
                    'numParameters': 0,
                    'trainable': layer_config.get('trainable', True)
                })
                
                # Create sequential connections
                if i > 0:
                    connections.append({
                        'source': f"layer_{i-1}",
                        'target': layer_id,
                        'tensorShape': None
                    })
        
        # Count parameters from model_weights
        if 'model_weights' in f:
            def count_h5_params(group):
                count = 0
                for key in group.keys():
                    item = group[key]
                    if isinstance(item, h5py.Dataset):
                        count += item.size
                    elif isinstance(item, h5py.Group):
                        count += count_h5_params(item)
                return count
            total_params = count_h5_params(f['model_weights'])
    
    architecture = {
        'name': Path(filename).stem,
        'framework': 'keras',
        'totalParameters': total_params,
        'trainableParameters': total_params,
        'inputShape': layers[0].get('inputShape') if layers else None,
        'outputShape': None,
        'layers': layers,
        'connections': connections
    }
    
    return AnalysisResponse(
        success=True,
        model_type='keras',
        architecture=architecture,
        message=f"Successfully analyzed Keras model with {len(layers)} layers"
    )


async def _analyze_tensorflow(file_path: str, filename: str) -> AnalysisResponse:
    """Analyze TensorFlow SavedModel or frozen graph."""
    try:
        import tensorflow as tf
    except ImportError:
        # Fallback: parse .pb file manually
        return await _analyze_pb_file(file_path, filename)
    
    layers = []
    connections = []
    
    # Try loading as SavedModel or GraphDef
    try:
        graph_def = tf.compat.v1.GraphDef()
        with open(file_path, 'rb') as f:
            graph_def.ParseFromString(f.read())
        
        node_map = {}
        for i, node in enumerate(graph_def.node):
            layer_id = f"layer_{i}"
            node_map[node.name] = layer_id
            
            # Extract attributes
            params = {}
            for key, attr in node.attr.items():
                if attr.HasField('i'):
                    params[key] = attr.i
                elif attr.HasField('f'):
                    params[key] = round(attr.f, 6)
                elif attr.HasField('s'):
                    params[key] = attr.s.decode('utf-8')
                elif attr.HasField('shape'):
                    dims = [d.size for d in attr.shape.dim]
                    params[key] = dims
            
            layers.append({
                'id': layer_id,
                'name': node.name,
                'type': node.op,
                'category': _infer_tf_category(node.op),
                'inputShape': None,
                'outputShape': None,
                'params': params,
                'numParameters': 0,
                'trainable': True
            })
            
            # Create connections from inputs
            for inp in node.input:
                inp_name = inp.lstrip('^').split(':')[0]
                if inp_name in node_map:
                    connections.append({
                        'source': node_map[inp_name],
                        'target': layer_id,
                        'tensorShape': None
                    })
        
        architecture = {
            'name': Path(filename).stem,
            'framework': 'tensorflow',
            'totalParameters': 0,
            'trainableParameters': 0,
            'inputShape': None,
            'outputShape': None,
            'layers': layers,
            'connections': connections
        }
        
        return AnalysisResponse(
            success=True,
            model_type='tensorflow_pb',
            architecture=architecture,
            message=f"Successfully analyzed TensorFlow graph with {len(layers)} nodes"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse TensorFlow model: {str(e)}")


async def _analyze_pb_file(file_path: str, filename: str) -> AnalysisResponse:
    """Fallback .pb file analyzer without TensorFlow."""
    raise HTTPException(
        status_code=501,
        detail="TensorFlow .pb analysis requires TensorFlow. Install with: pip install tensorflow"
    )


async def _analyze_safetensors(file_path: str, filename: str) -> AnalysisResponse:
    """Analyze SafeTensors file."""
    try:
        from safetensors import safe_open
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="safetensors not installed. Install with: pip install safetensors"
        )
    
    layers = []
    connections = []
    total_params = 0
    layer_groups = {}
    
    with safe_open(file_path, framework="pt") as f:
        tensor_names = list(f.keys())
        
        # Group tensors by layer
        for name in tensor_names:
            tensor = f.get_tensor(name)
            shape = list(tensor.shape)
            num_params = int(tensor.numel())
            total_params += num_params
            
            # Extract layer name from tensor name (e.g., "encoder.layer.0.attention.weight")
            parts = name.rsplit('.', 1)
            layer_name = parts[0] if len(parts) > 1 else name
            tensor_type = parts[1] if len(parts) > 1 else 'weight'
            
            if layer_name not in layer_groups:
                layer_groups[layer_name] = {
                    'tensors': {},
                    'params': {},
                    'total_params': 0
                }
            
            layer_groups[layer_name]['tensors'][tensor_type] = shape
            layer_groups[layer_name]['total_params'] += num_params
            
            # Infer params from shapes
            if tensor_type == 'weight' and len(shape) >= 2:
                layer_groups[layer_name]['params']['out_features'] = shape[0]
                layer_groups[layer_name]['params']['in_features'] = shape[1]
    
    # Convert groups to layers
    prev_layer_id = None
    for i, (layer_name, group) in enumerate(layer_groups.items()):
        layer_id = f"layer_{i}"
        
        # Infer layer type from name and shapes
        layer_type = _infer_layer_type_from_name(layer_name, group['tensors'])
        
        # Infer shapes
        input_shape = None
        output_shape = None
        if 'in_features' in group['params']:
            input_shape = [-1, group['params']['in_features']]
        if 'out_features' in group['params']:
            output_shape = [-1, group['params']['out_features']]
        
        layers.append({
            'id': layer_id,
            'name': layer_name,
            'type': layer_type,
            'category': _infer_category_from_type(layer_type),
            'inputShape': input_shape,
            'outputShape': output_shape,
            'params': group['params'],
            'numParameters': group['total_params'],
            'trainable': True
        })
        
        # Create sequential connections
        if prev_layer_id:
            connections.append({
                'source': prev_layer_id,
                'target': layer_id,
                'tensorShape': None
            })
        prev_layer_id = layer_id
    
    architecture = {
        'name': Path(filename).stem,
        'framework': 'safetensors',
        'totalParameters': total_params,
        'trainableParameters': total_params,
        'inputShape': layers[0].get('inputShape') if layers else None,
        'outputShape': layers[-1].get('outputShape') if layers else None,
        'layers': layers,
        'connections': connections
    }
    
    return AnalysisResponse(
        success=True,
        model_type='safetensors',
        architecture=architecture,
        message=f"Successfully analyzed SafeTensors model with {len(layers)} layers, {total_params:,} parameters"
    )


def _infer_keras_category(class_name: str) -> str:
    """Infer category from Keras layer class name."""
    name = class_name.lower()
    if 'conv' in name:
        return 'convolution'
    if 'pool' in name:
        return 'pooling'
    if 'dense' in name or 'linear' in name:
        return 'linear'
    if 'norm' in name or 'batch' in name:
        return 'normalization'
    if 'dropout' in name:
        return 'regularization'
    if 'lstm' in name or 'gru' in name or 'rnn' in name:
        return 'recurrent'
    if 'attention' in name:
        return 'attention'
    if 'activation' in name or 'relu' in name or 'sigmoid' in name:
        return 'activation'
    if 'embed' in name:
        return 'embedding'
    if 'flatten' in name or 'reshape' in name:
        return 'reshape'
    if 'input' in name:
        return 'input'
    return 'other'


def _infer_tf_category(op_type: str) -> str:
    """Infer category from TensorFlow op type."""
    op = op_type.lower()
    if 'conv' in op:
        return 'convolution'
    if 'pool' in op:
        return 'pooling'
    if 'matmul' in op or 'dense' in op:
        return 'linear'
    if 'norm' in op or 'batch' in op:
        return 'normalization'
    if 'relu' in op or 'sigmoid' in op or 'tanh' in op or 'softmax' in op:
        return 'activation'
    if 'placeholder' in op or 'input' in op:
        return 'input'
    if 'variable' in op or 'const' in op:
        return 'parameter'
    return 'other'


def _infer_layer_type_from_name(name: str, tensors: dict) -> str:
    """Infer layer type from name and tensor shapes."""
    name_lower = name.lower()
    
    if 'attention' in name_lower or 'attn' in name_lower:
        return 'MultiHeadAttention'
    if 'linear' in name_lower or 'dense' in name_lower or 'fc' in name_lower:
        return 'Linear'
    if 'conv' in name_lower:
        if 'weight' in tensors and len(tensors['weight']) == 4:
            return 'Conv2d'
        return 'Conv1d'
    if 'norm' in name_lower:
        if 'layer' in name_lower:
            return 'LayerNorm'
        return 'BatchNorm'
    if 'embed' in name_lower:
        return 'Embedding'
    if 'lstm' in name_lower:
        return 'LSTM'
    if 'gru' in name_lower:
        return 'GRU'
    if 'query' in name_lower or 'key' in name_lower or 'value' in name_lower:
        return 'Linear'
    
    # Infer from tensor shapes
    if 'weight' in tensors:
        shape = tensors['weight']
        if len(shape) == 2:
            return 'Linear'
        if len(shape) == 4:
            return 'Conv2d'
        if len(shape) == 1:
            return 'LayerNorm'
    
    return 'Unknown'


def _infer_category_from_type(layer_type: str) -> str:
    """Infer category from layer type."""
    type_lower = layer_type.lower()
    if 'conv' in type_lower:
        return 'convolution'
    if 'linear' in type_lower:
        return 'linear'
    if 'norm' in type_lower:
        return 'normalization'
    if 'attention' in type_lower:
        return 'attention'
    if 'embed' in type_lower:
        return 'embedding'
    if 'lstm' in type_lower or 'gru' in type_lower or 'rnn' in type_lower:
        return 'recurrent'
    return 'other'


# =============================================================================
# Saved Models API Endpoints
# =============================================================================

class SaveModelRequest(BaseModel):
    """Request to save a model."""
    name: str
    framework: str
    totalParameters: int
    layerCount: int
    architecture: dict
    fileHash: Optional[str] = None


class SavedModelSummary(BaseModel):
    """Summary of a saved model."""
    id: int
    name: str
    framework: str
    total_parameters: int
    layer_count: int
    created_at: str


@app.get("/models/saved")
async def list_saved_models():
    """
    Get a list of all saved models.
    Returns metadata only (not full architecture).
    """
    try:
        models = db.get_saved_models()
        return {
            "success": True,
            "models": models
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/saved/{model_id}")
async def get_saved_model(model_id: int):
    """
    Get a saved model by ID with full architecture.
    """
    try:
        model = db.get_model_by_id(model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return {
            "success": True,
            "model": model
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/save")
async def save_model(request: SaveModelRequest):
    """
    Save a processed model to the database.
    """
    try:
        model_id = db.save_model(
            name=request.name,
            framework=request.framework,
            total_parameters=request.totalParameters,
            layer_count=request.layerCount,
            architecture=request.architecture,
            file_hash=request.fileHash
        )
        
        return {
            "success": True,
            "id": model_id,
            "message": "Model saved successfully"
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/models/saved/{model_id}")
async def delete_saved_model(model_id: int):
    """
    Delete a saved model by ID.
    """
    try:
        deleted = db.delete_model(model_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return {
            "success": True,
            "message": "Model deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
