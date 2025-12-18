"""
PyTorch Model Analyzer
Extracts architecture information from PyTorch models for 3D visualization.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import OrderedDict
import json


@dataclass
class LayerInfo:
    """Information about a single layer in the model."""
    id: str
    name: str
    type: str
    category: str
    input_shape: Optional[List[int]]
    output_shape: Optional[List[int]]
    params: Dict[str, Any]
    num_parameters: int
    trainable: bool


@dataclass
class ConnectionInfo:
    """Information about connections between layers."""
    source: str
    target: str
    tensor_shape: Optional[List[int]]


@dataclass
class ModelArchitecture:
    """Complete model architecture information."""
    name: str
    framework: str
    total_parameters: int
    trainable_parameters: int
    layers: List[LayerInfo]
    connections: List[ConnectionInfo]
    input_shape: Optional[List[int]]
    output_shape: Optional[List[int]]


# Layer category mapping
LAYER_CATEGORIES = {
    # Convolution layers
    'Conv1d': 'convolution',
    'Conv2d': 'convolution',
    'Conv3d': 'convolution',
    'ConvTranspose1d': 'convolution',
    'ConvTranspose2d': 'convolution',
    'ConvTranspose3d': 'convolution',
    
    # Pooling layers
    'MaxPool1d': 'pooling',
    'MaxPool2d': 'pooling',
    'MaxPool3d': 'pooling',
    'AvgPool1d': 'pooling',
    'AvgPool2d': 'pooling',
    'AvgPool3d': 'pooling',
    'AdaptiveAvgPool1d': 'pooling',
    'AdaptiveAvgPool2d': 'pooling',
    'AdaptiveAvgPool3d': 'pooling',
    'AdaptiveMaxPool1d': 'pooling',
    'AdaptiveMaxPool2d': 'pooling',
    'AdaptiveMaxPool3d': 'pooling',
    'GlobalAveragePooling2D': 'pooling',
    
    # Linear/Dense layers
    'Linear': 'linear',
    'LazyLinear': 'linear',
    'Bilinear': 'linear',
    
    # Normalization layers
    'BatchNorm1d': 'normalization',
    'BatchNorm2d': 'normalization',
    'BatchNorm3d': 'normalization',
    'LayerNorm': 'normalization',
    'GroupNorm': 'normalization',
    'InstanceNorm1d': 'normalization',
    'InstanceNorm2d': 'normalization',
    'InstanceNorm3d': 'normalization',
    
    # Activation layers
    'ReLU': 'activation',
    'ReLU6': 'activation',
    'LeakyReLU': 'activation',
    'PReLU': 'activation',
    'ELU': 'activation',
    'SELU': 'activation',
    'GELU': 'activation',
    'Sigmoid': 'activation',
    'Tanh': 'activation',
    'Softmax': 'activation',
    'LogSoftmax': 'activation',
    'Softplus': 'activation',
    'Softsign': 'activation',
    'Hardswish': 'activation',
    'Hardsigmoid': 'activation',
    'SiLU': 'activation',
    'Mish': 'activation',
    
    # Dropout layers
    'Dropout': 'regularization',
    'Dropout2d': 'regularization',
    'Dropout3d': 'regularization',
    'AlphaDropout': 'regularization',
    
    # Recurrent layers
    'RNN': 'recurrent',
    'LSTM': 'recurrent',
    'GRU': 'recurrent',
    'RNNCell': 'recurrent',
    'LSTMCell': 'recurrent',
    'GRUCell': 'recurrent',
    
    # Transformer layers
    'Transformer': 'attention',
    'TransformerEncoder': 'attention',
    'TransformerDecoder': 'attention',
    'TransformerEncoderLayer': 'attention',
    'TransformerDecoderLayer': 'attention',
    'MultiheadAttention': 'attention',
    
    # Embedding layers
    'Embedding': 'embedding',
    'EmbeddingBag': 'embedding',
    
    # Reshape/View layers
    'Flatten': 'reshape',
    'Unflatten': 'reshape',
    
    # Container layers
    'Sequential': 'container',
    'ModuleList': 'container',
    'ModuleDict': 'container',
}


def get_layer_category(layer_type: str) -> str:
    """Get the category for a layer type."""
    return LAYER_CATEGORIES.get(layer_type, 'other')


def count_parameters(module: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in a module."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def extract_layer_params(module: nn.Module, layer_type: str) -> Dict[str, Any]:
    """Extract relevant parameters from a layer."""
    params = {}
    
    try:
        if hasattr(module, 'in_features'):
            params['in_features'] = module.in_features
        if hasattr(module, 'out_features'):
            params['out_features'] = module.out_features
        if hasattr(module, 'in_channels'):
            params['in_channels'] = module.in_channels
        if hasattr(module, 'out_channels'):
            params['out_channels'] = module.out_channels
        if hasattr(module, 'kernel_size'):
            ks = module.kernel_size
            params['kernel_size'] = list(ks) if isinstance(ks, tuple) else ks
        if hasattr(module, 'stride'):
            s = module.stride
            params['stride'] = list(s) if isinstance(s, tuple) else s
        if hasattr(module, 'padding'):
            p = module.padding
            params['padding'] = list(p) if isinstance(p, tuple) else p
        if hasattr(module, 'dilation'):
            d = module.dilation
            params['dilation'] = list(d) if isinstance(d, tuple) else d
        if hasattr(module, 'groups'):
            params['groups'] = module.groups
        if hasattr(module, 'bias') and module.bias is not None:
            params['bias'] = True
        if hasattr(module, 'num_features'):
            params['num_features'] = module.num_features
        if hasattr(module, 'eps'):
            params['eps'] = module.eps
        if hasattr(module, 'momentum') and module.momentum is not None:
            params['momentum'] = module.momentum
        if hasattr(module, 'normalized_shape'):
            params['normalized_shape'] = list(module.normalized_shape)
        if hasattr(module, 'hidden_size'):
            params['hidden_size'] = module.hidden_size
        if hasattr(module, 'num_layers'):
            params['num_layers'] = module.num_layers
        if hasattr(module, 'bidirectional'):
            params['bidirectional'] = module.bidirectional
        if hasattr(module, 'num_heads'):
            params['num_heads'] = module.num_heads
        if hasattr(module, 'embed_dim'):
            params['embed_dim'] = module.embed_dim
        if hasattr(module, 'num_embeddings'):
            params['num_embeddings'] = module.num_embeddings
        if hasattr(module, 'embedding_dim'):
            params['embedding_dim'] = module.embedding_dim
        if hasattr(module, 'p') and layer_type.startswith('Dropout'):
            params['p'] = module.p
        if hasattr(module, 'negative_slope'):
            params['negative_slope'] = module.negative_slope
        if hasattr(module, 'inplace'):
            params['inplace'] = module.inplace
        if hasattr(module, 'dim'):
            params['dim'] = module.dim
    except Exception:
        pass
    
    return params


def analyze_model_structure(model: nn.Module, model_name: str = "model") -> ModelArchitecture:
    """
    Analyze a PyTorch model and extract its architecture.
    
    Args:
        model: The PyTorch model to analyze
        model_name: Name identifier for the model
        
    Returns:
        ModelArchitecture with complete layer and connection information
    """
    layers = []
    connections = []
    layer_index = 0
    parent_stack = []
    
    def process_module(name: str, module: nn.Module, parent_id: Optional[str] = None):
        nonlocal layer_index
        
        layer_type = module.__class__.__name__
        
        # Skip container modules but process their children
        if layer_type in ('Sequential', 'ModuleList', 'ModuleDict'):
            for child_name, child in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                process_module(full_name, child, parent_id)
            return
        
        # Skip modules with no parameters and no meaningful operation
        # But include activation, pooling, dropout, etc.
        has_params = sum(1 for _ in module.parameters(recurse=False)) > 0
        is_meaningful = layer_type in LAYER_CATEGORIES or has_params
        
        if not is_meaningful and len(list(module.children())) > 0:
            # Process children of non-meaningful containers
            for child_name, child in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                process_module(full_name, child, parent_id)
            return
        
        layer_id = f"layer_{layer_index}"
        layer_index += 1
        
        total_params, trainable_params = count_parameters(module)
        params = extract_layer_params(module, layer_type)
        
        layer_info = LayerInfo(
            id=layer_id,
            name=name or layer_type,
            type=layer_type,
            category=get_layer_category(layer_type),
            input_shape=None,  # Will be populated during forward pass
            output_shape=None,
            params=params,
            num_parameters=total_params,
            trainable=trainable_params > 0
        )
        
        layers.append(layer_info)
        
        # Create connection from parent
        if parent_id is not None:
            connections.append(ConnectionInfo(
                source=parent_id,
                target=layer_id,
                tensor_shape=None
            ))
        
        # Process children
        children = list(module.named_children())
        if children:
            for child_name, child in children:
                full_name = f"{name}.{child_name}" if name else child_name
                process_module(full_name, child, layer_id)
        
        return layer_id
    
    # Process the model
    children = list(model.named_children())
    if children:
        prev_id = None
        for name, child in children:
            layer_id = process_module(name, child, prev_id)
            if layer_id:
                prev_id = layer_id
    else:
        # Single layer model
        process_module("", model, None)
    
    # If layers are sequential and no connections exist, create linear connections
    if len(layers) > 1 and len(connections) == 0:
        for i in range(len(layers) - 1):
            connections.append(ConnectionInfo(
                source=layers[i].id,
                target=layers[i + 1].id,
                tensor_shape=None
            ))
    
    total_params, trainable_params = count_parameters(model)
    
    return ModelArchitecture(
        name=model_name,
        framework="pytorch",
        total_parameters=total_params,
        trainable_parameters=trainable_params,
        layers=layers,
        connections=connections,
        input_shape=None,
        output_shape=None
    )


def trace_model_shapes(model: nn.Module, input_tensor: torch.Tensor, arch: ModelArchitecture) -> ModelArchitecture:
    """
    Trace model execution to capture input/output shapes for each layer.
    
    Args:
        model: The PyTorch model
        input_tensor: Sample input tensor
        arch: Existing architecture info to update
        
    Returns:
        Updated ModelArchitecture with shape information
    """
    shapes = {}
    hooks = []
    
    def make_hook(name):
        def hook(module, input, output):
            input_shape = None
            output_shape = None
            
            if isinstance(input, tuple) and len(input) > 0:
                if isinstance(input[0], torch.Tensor):
                    input_shape = list(input[0].shape)
            elif isinstance(input, torch.Tensor):
                input_shape = list(input.shape)
            
            if isinstance(output, torch.Tensor):
                output_shape = list(output.shape)
            elif isinstance(output, tuple) and len(output) > 0:
                if isinstance(output[0], torch.Tensor):
                    output_shape = list(output[0].shape)
            
            shapes[name] = {
                'input': input_shape,
                'output': output_shape
            }
        return hook
    
    # Register hooks
    for name, module in model.named_modules():
        if name:  # Skip root module
            hooks.append(module.register_forward_hook(make_hook(name)))
    
    # Run forward pass
    try:
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        # Update architecture with shapes
        for layer in arch.layers:
            if layer.name in shapes:
                layer.input_shape = shapes[layer.name]['input']
                layer.output_shape = shapes[layer.name]['output']
        
        # Set model input/output shapes
        arch.input_shape = list(input_tensor.shape)
        if isinstance(output, torch.Tensor):
            arch.output_shape = list(output.shape)
        
    except Exception as e:
        print(f"Warning: Could not trace shapes: {e}")
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    return arch


def load_pytorch_model(file_path: str) -> Tuple[Optional[nn.Module], Optional[Dict], str]:
    """
    Load a PyTorch model from file.
    
    Returns:
        Tuple of (model, state_dict, model_type)
        model_type can be: 'full_model', 'state_dict', 'torchscript', 'checkpoint'
    """
    try:
        # Try loading as TorchScript first
        try:
            model = torch.jit.load(file_path, map_location='cpu')
            return model, None, 'torchscript'
        except Exception:
            pass
        
        # Try loading as regular checkpoint
        checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, nn.Module):
            return checkpoint, None, 'full_model'
        
        if isinstance(checkpoint, dict):
            # Check for common checkpoint formats
            if 'model' in checkpoint:
                if isinstance(checkpoint['model'], nn.Module):
                    return checkpoint['model'], None, 'checkpoint'
                elif isinstance(checkpoint['model'], dict):
                    return None, checkpoint['model'], 'state_dict'
            
            if 'state_dict' in checkpoint:
                return None, checkpoint['state_dict'], 'state_dict'
            
            if 'model_state_dict' in checkpoint:
                return None, checkpoint['model_state_dict'], 'state_dict'
            
            # Check if it's directly a state dict (contains tensor values)
            has_tensors = any(isinstance(v, torch.Tensor) for v in checkpoint.values())
            if has_tensors:
                return None, checkpoint, 'state_dict'
        
        return None, None, 'unknown'
        
    except Exception as e:
        raise ValueError(f"Failed to load model: {str(e)}")


def analyze_state_dict(state_dict: Dict[str, torch.Tensor], model_name: str = "model") -> ModelArchitecture:
    """
    Analyze a state dict to infer model architecture.
    
    This extracts layer information from weight tensor names and shapes.
    """
    layers = []
    layer_map = OrderedDict()
    
    # Group parameters by layer name
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        
        # Extract layer name from parameter name
        parts = key.rsplit('.', 1)
        if len(parts) == 2:
            layer_name, param_type = parts
        else:
            layer_name = key
            param_type = 'weight'
        
        if layer_name not in layer_map:
            layer_map[layer_name] = {
                'params': {},
                'shapes': {}
            }
        
        layer_map[layer_name]['params'][param_type] = True
        layer_map[layer_name]['shapes'][param_type] = list(tensor.shape)
    
    # Create layer info from grouped parameters
    layer_index = 0
    for layer_name, info in layer_map.items():
        layer_type, category = infer_layer_type(layer_name, info['shapes'])
        
        layer_id = f"layer_{layer_index}"
        layer_index += 1
        
        # Compute number of parameters
        num_params = 0
        for param_type, shape in info['shapes'].items():
            param_size = 1
            for dim in shape:
                param_size *= dim
            num_params += param_size
        
        # Extract layer parameters from shapes
        params = extract_params_from_shapes(layer_type, info['shapes'])
        
        # Infer input/output shapes
        input_shape, output_shape = infer_shapes(layer_type, info['shapes'], params)
        
        layers.append(LayerInfo(
            id=layer_id,
            name=layer_name,
            type=layer_type,
            category=category,
            input_shape=input_shape,
            output_shape=output_shape,
            params=params,
            num_parameters=num_params,
            trainable=True
        ))
    
    # Create sequential connections
    connections = []
    for i in range(len(layers) - 1):
        connections.append(ConnectionInfo(
            source=layers[i].id,
            target=layers[i + 1].id,
            tensor_shape=layers[i].output_shape
        ))
    
    total_params = sum(layer.num_parameters for layer in layers)
    
    return ModelArchitecture(
        name=model_name,
        framework="pytorch",
        total_parameters=total_params,
        trainable_parameters=total_params,
        layers=layers,
        connections=connections,
        input_shape=layers[0].input_shape if layers else None,
        output_shape=layers[-1].output_shape if layers else None
    )


def infer_layer_type(layer_name: str, shapes: Dict[str, List[int]]) -> Tuple[str, str]:
    """Infer layer type from name and weight shapes."""
    name_lower = layer_name.lower()
    
    # Check for common layer type patterns in name
    if 'conv' in name_lower:
        weight_shape = shapes.get('weight', [])
        if len(weight_shape) == 5:
            return 'Conv3d', 'convolution'
        elif len(weight_shape) == 4:
            return 'Conv2d', 'convolution'
        elif len(weight_shape) == 3:
            return 'Conv1d', 'convolution'
        return 'Conv2d', 'convolution'
    
    if 'bn' in name_lower or 'batch' in name_lower or 'norm' in name_lower:
        weight_shape = shapes.get('weight', shapes.get('running_mean', []))
        if 'layer' in name_lower:
            return 'LayerNorm', 'normalization'
        return 'BatchNorm2d', 'normalization'
    
    if 'fc' in name_lower or 'linear' in name_lower or 'dense' in name_lower or 'classifier' in name_lower:
        return 'Linear', 'linear'
    
    if 'lstm' in name_lower:
        return 'LSTM', 'recurrent'
    
    if 'gru' in name_lower:
        return 'GRU', 'recurrent'
    
    if 'rnn' in name_lower:
        return 'RNN', 'recurrent'
    
    if 'attention' in name_lower or 'attn' in name_lower:
        return 'MultiheadAttention', 'attention'
    
    if 'embed' in name_lower:
        return 'Embedding', 'embedding'
    
    if 'pool' in name_lower:
        return 'AdaptiveAvgPool2d', 'pooling'
    
    # Infer from weight shape
    weight_shape = shapes.get('weight', [])
    if len(weight_shape) == 2:
        return 'Linear', 'linear'
    elif len(weight_shape) == 4:
        return 'Conv2d', 'convolution'
    elif len(weight_shape) == 3:
        return 'Conv1d', 'convolution'
    elif len(weight_shape) == 1:
        return 'BatchNorm2d', 'normalization'
    
    return 'Unknown', 'other'


def extract_params_from_shapes(layer_type: str, shapes: Dict[str, List[int]]) -> Dict[str, Any]:
    """Extract layer parameters from weight shapes."""
    params = {}
    weight_shape = shapes.get('weight', [])
    
    if layer_type in ('Linear',):
        if len(weight_shape) >= 2:
            params['out_features'] = weight_shape[0]
            params['in_features'] = weight_shape[1]
            params['bias'] = 'bias' in shapes
    
    elif layer_type in ('Conv1d', 'Conv2d', 'Conv3d'):
        if len(weight_shape) >= 2:
            params['out_channels'] = weight_shape[0]
            params['in_channels'] = weight_shape[1]
            if len(weight_shape) > 2:
                params['kernel_size'] = weight_shape[2:]
            params['bias'] = 'bias' in shapes
    
    elif layer_type in ('BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d'):
        if len(weight_shape) >= 1:
            params['num_features'] = weight_shape[0]
    
    elif layer_type == 'LayerNorm':
        if len(weight_shape) >= 1:
            params['normalized_shape'] = weight_shape
    
    elif layer_type == 'Embedding':
        if len(weight_shape) >= 2:
            params['num_embeddings'] = weight_shape[0]
            params['embedding_dim'] = weight_shape[1]
    
    elif layer_type in ('LSTM', 'GRU', 'RNN'):
        # weight_ih_l0 shape gives hidden_size x input_size
        if 'weight_ih_l0' in shapes:
            ih_shape = shapes['weight_ih_l0']
            if len(ih_shape) >= 2:
                multiplier = 4 if layer_type == 'LSTM' else (3 if layer_type == 'GRU' else 1)
                params['hidden_size'] = ih_shape[0] // multiplier
                params['input_size'] = ih_shape[1]
    
    return params


def infer_shapes(layer_type: str, shapes: Dict[str, List[int]], params: Dict[str, Any]) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    """Infer input/output shapes from layer parameters."""
    input_shape = None
    output_shape = None
    
    if layer_type == 'Linear':
        if 'in_features' in params:
            input_shape = [-1, params['in_features']]
        if 'out_features' in params:
            output_shape = [-1, params['out_features']]
    
    elif layer_type in ('Conv2d',):
        if 'in_channels' in params:
            input_shape = [-1, params['in_channels'], -1, -1]
        if 'out_channels' in params:
            output_shape = [-1, params['out_channels'], -1, -1]
    
    elif layer_type in ('Conv1d',):
        if 'in_channels' in params:
            input_shape = [-1, params['in_channels'], -1]
        if 'out_channels' in params:
            output_shape = [-1, params['out_channels'], -1]
    
    elif layer_type in ('BatchNorm2d',):
        if 'num_features' in params:
            input_shape = [-1, params['num_features'], -1, -1]
            output_shape = [-1, params['num_features'], -1, -1]
    
    elif layer_type == 'Embedding':
        if 'embedding_dim' in params:
            output_shape = [-1, -1, params['embedding_dim']]
    
    elif layer_type in ('GRU', 'LSTM', 'RNN'):
        # For recurrent layers: input is (batch, seq_len, input_size)
        # output is (batch, seq_len, hidden_size * num_directions)
        if 'input_size' in params:
            input_shape = [-1, -1, params['input_size']]
        if 'hidden_size' in params:
            num_directions = 2 if params.get('bidirectional', False) else 1
            output_shape = [-1, -1, params['hidden_size'] * num_directions]
    
    return input_shape, output_shape


def architecture_to_dict(arch: ModelArchitecture) -> Dict[str, Any]:
    """Convert ModelArchitecture to JSON-serializable dict."""
    return {
        'name': arch.name,
        'framework': arch.framework,
        'totalParameters': arch.total_parameters,
        'trainableParameters': arch.trainable_parameters,
        'inputShape': arch.input_shape,
        'outputShape': arch.output_shape,
        'layers': [
            {
                'id': layer.id,
                'name': layer.name,
                'type': layer.type,
                'category': layer.category,
                'inputShape': layer.input_shape,
                'outputShape': layer.output_shape,
                'params': layer.params,
                'numParameters': layer.num_parameters,
                'trainable': layer.trainable
            }
            for layer in arch.layers
        ],
        'connections': [
            {
                'source': conn.source,
                'target': conn.target,
                'tensorShape': conn.tensor_shape
            }
            for conn in arch.connections
        ]
    }
