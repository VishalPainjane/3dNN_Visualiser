"""
PyTorch Model Exporter

Export PyTorch models to .nn3d format for visualization.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from collections import OrderedDict

from .schema import (
    NN3DModel, NN3DGraph, NN3DNode, NN3DEdge, NN3DMetadata,
    NN3DSubgraph, LayerParams, LayerType, VisualizationConfig
)


# Mapping from PyTorch module types to NN3D layer types
PYTORCH_TO_NN3D_TYPE: Dict[type, str] = {
    # Convolution layers
    nn.Conv1d: LayerType.CONV1D.value,
    nn.Conv2d: LayerType.CONV2D.value,
    nn.Conv3d: LayerType.CONV3D.value,
    nn.ConvTranspose2d: LayerType.CONV_TRANSPOSE_2D.value,
    
    # Linear layers
    nn.Linear: LayerType.LINEAR.value,
    nn.Embedding: LayerType.EMBEDDING.value,
    
    # Normalization layers
    nn.BatchNorm1d: LayerType.BATCH_NORM_1D.value,
    nn.BatchNorm2d: LayerType.BATCH_NORM_2D.value,
    nn.LayerNorm: LayerType.LAYER_NORM.value,
    nn.GroupNorm: LayerType.GROUP_NORM.value,
    nn.InstanceNorm2d: LayerType.INSTANCE_NORM.value,
    nn.Dropout: LayerType.DROPOUT.value,
    nn.Dropout2d: LayerType.DROPOUT.value,
    
    # Activation layers
    nn.ReLU: LayerType.RELU.value,
    nn.LeakyReLU: LayerType.LEAKY_RELU.value,
    nn.GELU: LayerType.GELU.value,
    nn.SiLU: LayerType.SILU.value,
    nn.Sigmoid: LayerType.SIGMOID.value,
    nn.Tanh: LayerType.TANH.value,
    nn.Softmax: LayerType.SOFTMAX.value,
    
    # Pooling layers
    nn.MaxPool1d: LayerType.MAX_POOL_1D.value,
    nn.MaxPool2d: LayerType.MAX_POOL_2D.value,
    nn.AvgPool2d: LayerType.AVG_POOL_2D.value,
    nn.AdaptiveAvgPool2d: LayerType.ADAPTIVE_AVG_POOL.value,
    nn.AdaptiveAvgPool1d: LayerType.ADAPTIVE_AVG_POOL.value,
    
    # Recurrent layers
    nn.LSTM: LayerType.LSTM.value,
    nn.GRU: LayerType.GRU.value,
    nn.RNN: LayerType.RNN.value,
    
    # Attention layers
    nn.MultiheadAttention: LayerType.MULTI_HEAD_ATTENTION.value,
    
    # Transform layers
    nn.Flatten: LayerType.FLATTEN.value,
    nn.Upsample: LayerType.UPSAMPLE.value,
}


class PyTorchExporter:
    """
    Export PyTorch models to NN3D format.
    
    Usage:
        exporter = PyTorchExporter(model, input_shape=(1, 3, 224, 224))
        nn3d_model = exporter.export()
        nn3d_model.save("model.nn3d")
    """
    
    def __init__(
        self,
        model: nn.Module,
        input_shape: Optional[Tuple[int, ...]] = None,
        model_name: Optional[str] = None,
        include_activations: bool = False,
    ):
        """
        Initialize the exporter.
        
        Args:
            model: PyTorch model to export
            input_shape: Input tensor shape (batch, channels, height, width)
            model_name: Name for the model (defaults to class name)
            include_activations: Whether to trace activations (requires input_shape)
        """
        self.model = model
        self.input_shape = input_shape
        self.model_name = model_name or model.__class__.__name__
        self.include_activations = include_activations
        
        self.nodes: List[NN3DNode] = []
        self.edges: List[NN3DEdge] = []
        self.subgraphs: List[NN3DSubgraph] = []
        
        self._node_id_counter = 0
        self._module_to_id: Dict[nn.Module, str] = {}
        self._shapes: Dict[str, Tuple[List[int], List[int]]] = {}
    
    def _get_node_id(self) -> str:
        """Generate unique node ID"""
        node_id = f"node_{self._node_id_counter}"
        self._node_id_counter += 1
        return node_id
    
    def _get_layer_type(self, module: nn.Module) -> str:
        """Map PyTorch module to NN3D layer type"""
        module_type = type(module)
        
        if module_type in PYTORCH_TO_NN3D_TYPE:
            return PYTORCH_TO_NN3D_TYPE[module_type]
        
        # Check for common container patterns
        class_name = module_type.__name__.lower()
        
        if 'attention' in class_name:
            return LayerType.ATTENTION.value
        elif 'transformer' in class_name:
            return LayerType.TRANSFORMER.value
        elif 'encoder' in class_name:
            return LayerType.ENCODER_BLOCK.value
        elif 'decoder' in class_name:
            return LayerType.DECODER_BLOCK.value
        elif 'residual' in class_name or 'resblock' in class_name:
            return LayerType.RESIDUAL_BLOCK.value
        
        return LayerType.CUSTOM.value
    
    def _extract_params(self, module: nn.Module) -> LayerParams:
        """Extract layer parameters from PyTorch module"""
        params = LayerParams()
        
        # Convolution parameters
        if hasattr(module, 'in_channels'):
            params.in_channels = module.in_channels
        if hasattr(module, 'out_channels'):
            params.out_channels = module.out_channels
        if hasattr(module, 'kernel_size'):
            ks = module.kernel_size
            params.kernel_size = list(ks) if isinstance(ks, tuple) else ks
        if hasattr(module, 'stride'):
            stride = module.stride
            params.stride = list(stride) if isinstance(stride, tuple) else stride
        if hasattr(module, 'padding'):
            pad = module.padding
            params.padding = list(pad) if isinstance(pad, tuple) else pad
        if hasattr(module, 'dilation'):
            dil = module.dilation
            params.dilation = list(dil) if isinstance(dil, tuple) else dil
        if hasattr(module, 'groups'):
            params.groups = module.groups
        
        # Linear parameters
        if hasattr(module, 'in_features'):
            params.in_features = module.in_features
        if hasattr(module, 'out_features'):
            params.out_features = module.out_features
        
        # Attention parameters
        if hasattr(module, 'num_heads'):
            params.num_heads = module.num_heads
        if hasattr(module, 'embed_dim'):
            params.hidden_size = module.embed_dim
        
        # Normalization parameters
        if hasattr(module, 'eps'):
            params.eps = module.eps
        if hasattr(module, 'momentum') and module.momentum is not None:
            params.momentum = module.momentum
        if hasattr(module, 'affine'):
            params.affine = module.affine
        
        # Dropout parameters
        if hasattr(module, 'p'):
            params.dropout_rate = module.p
        
        # Embedding parameters
        if hasattr(module, 'num_embeddings'):
            params.num_embeddings = module.num_embeddings
        if hasattr(module, 'embedding_dim'):
            params.embedding_dim = module.embedding_dim
        
        # Bias
        if hasattr(module, 'bias') and module.bias is not None:
            params.bias = True
        elif hasattr(module, 'bias'):
            params.bias = False
        
        return params
    
    def _trace_shapes(self) -> None:
        """Trace tensor shapes through the model"""
        if self.input_shape is None:
            return
        
        hooks = []
        
        def hook_fn(name: str):
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
                
                self._shapes[name] = (input_shape, output_shape)
            return hook
        
        # Register hooks
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Run forward pass
        try:
            self.model.eval()
            with torch.no_grad():
                dummy_input = torch.zeros(self.input_shape)
                self.model(dummy_input)
        except Exception as e:
            print(f"Warning: Could not trace shapes: {e}")
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
    
    def _process_module(
        self,
        module: nn.Module,
        name: str,
        parent_id: Optional[str] = None,
        depth: int = 0
    ) -> Optional[str]:
        """Process a single module and its children"""
        
        # Skip container modules without parameters
        children = list(module.named_children())
        is_leaf = len(children) == 0
        
        # Create node for leaf modules or significant containers
        layer_type = self._get_layer_type(module)
        
        if is_leaf or layer_type != LayerType.CUSTOM.value:
            node_id = self._get_node_id()
            self._module_to_id[module] = node_id
            
            # Get shapes if traced
            input_shape, output_shape = self._shapes.get(name, (None, None))
            
            # Extract parameters
            params = self._extract_params(module)
            
            # Count parameters
            num_params = sum(p.numel() for p in module.parameters(recurse=False))
            
            node = NN3DNode(
                id=node_id,
                type=layer_type,
                name=name or module.__class__.__name__,
                params=params if any(v is not None for v in [
                    params.in_channels, params.out_channels, 
                    params.in_features, params.out_features,
                    params.kernel_size, params.num_heads
                ]) else None,
                input_shape=input_shape,
                output_shape=output_shape,
                depth=depth,
                attributes={'num_params': num_params} if num_params > 0 else None
            )
            
            self.nodes.append(node)
            
            # Create edge from parent
            if parent_id:
                edge = NN3DEdge(
                    source=parent_id,
                    target=node_id,
                    tensor_shape=input_shape,
                )
                self.edges.append(edge)
            
            # Process children
            if children:
                subgraph_nodes = [node_id]
                prev_id = node_id
                
                for child_name, child in children:
                    full_name = f"{name}.{child_name}" if name else child_name
                    child_id = self._process_module(child, full_name, prev_id, depth + 1)
                    if child_id:
                        subgraph_nodes.append(child_id)
                        prev_id = child_id
                
                # Create subgraph for container
                if len(subgraph_nodes) > 1:
                    self.subgraphs.append(NN3DSubgraph(
                        id=f"subgraph_{node_id}",
                        name=name or module.__class__.__name__,
                        nodes=subgraph_nodes,
                        type='sequential'
                    ))
                
                return prev_id
            
            return node_id
        
        # Process children of container
        prev_id = parent_id
        for child_name, child in children:
            full_name = f"{name}.{child_name}" if name else child_name
            child_id = self._process_module(child, full_name, prev_id, depth)
            if child_id:
                prev_id = child_id
        
        return prev_id
    
    def export(self) -> NN3DModel:
        """Export the model to NN3D format"""
        
        # Trace shapes first
        self._trace_shapes()
        
        # Add input node
        input_id = self._get_node_id()
        input_node = NN3DNode(
            id=input_id,
            type=LayerType.INPUT.value,
            name="input",
            output_shape=list(self.input_shape) if self.input_shape else None,
            depth=0
        )
        self.nodes.append(input_node)
        
        # Process all modules
        last_id = self._process_module(self.model, "", input_id, 1)
        
        # Add output node
        output_id = self._get_node_id()
        output_shape = None
        if last_id and self.nodes:
            # Get output shape from last processed node
            for node in reversed(self.nodes):
                if node.output_shape:
                    output_shape = node.output_shape
                    break
        
        output_node = NN3DNode(
            id=output_id,
            type=LayerType.OUTPUT.value,
            name="output",
            input_shape=output_shape,
            depth=len(self.nodes)
        )
        self.nodes.append(output_node)
        
        if last_id:
            self.edges.append(NN3DEdge(
                source=last_id,
                target=output_id,
                tensor_shape=output_shape
            ))
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Create metadata
        metadata = NN3DMetadata(
            name=self.model_name,
            framework="pytorch",
            created=datetime.now().isoformat(),
            input_shape=list(self.input_shape) if self.input_shape else None,
            output_shape=output_shape,
            total_params=total_params,
            trainable_params=trainable_params,
        )
        
        # Create graph
        graph = NN3DGraph(
            nodes=self.nodes,
            edges=self.edges,
            subgraphs=self.subgraphs if self.subgraphs else None
        )
        
        # Create visualization config
        viz_config = VisualizationConfig()
        
        return NN3DModel(
            metadata=metadata,
            graph=graph,
            visualization=viz_config
        )


def export_pytorch_model(
    model: nn.Module,
    output_path: str,
    input_shape: Optional[Tuple[int, ...]] = None,
    model_name: Optional[str] = None,
) -> NN3DModel:
    """
    Convenience function to export a PyTorch model to .nn3d file.
    
    Args:
        model: PyTorch model to export
        output_path: Path to save the .nn3d file
        input_shape: Input tensor shape (batch, channels, height, width)
        model_name: Name for the model
        
    Returns:
        The exported NN3DModel
    """
    exporter = PyTorchExporter(model, input_shape, model_name)
    nn3d_model = exporter.export()
    nn3d_model.save(output_path)
    return nn3d_model
