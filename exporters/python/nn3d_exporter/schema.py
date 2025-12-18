"""
NN3D Schema definitions using Python dataclasses

These classes mirror the JSON schema and can be serialized to .nn3d format.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import json
from datetime import datetime


class LayerType(str, Enum):
    """Supported layer types"""
    INPUT = "input"
    OUTPUT = "output"
    CONV1D = "conv1d"
    CONV2D = "conv2d"
    CONV3D = "conv3d"
    CONV_TRANSPOSE_2D = "convTranspose2d"
    DEPTHWISE_CONV2D = "depthwiseConv2d"
    SEPARABLE_CONV2D = "separableConv2d"
    LINEAR = "linear"
    DENSE = "dense"
    EMBEDDING = "embedding"
    BATCH_NORM_1D = "batchNorm1d"
    BATCH_NORM_2D = "batchNorm2d"
    LAYER_NORM = "layerNorm"
    GROUP_NORM = "groupNorm"
    INSTANCE_NORM = "instanceNorm"
    DROPOUT = "dropout"
    RELU = "relu"
    LEAKY_RELU = "leakyRelu"
    GELU = "gelu"
    SILU = "silu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"
    MAX_POOL_1D = "maxPool1d"
    MAX_POOL_2D = "maxPool2d"
    AVG_POOL_2D = "avgPool2d"
    GLOBAL_AVG_POOL = "globalAvgPool"
    ADAPTIVE_AVG_POOL = "adaptiveAvgPool"
    FLATTEN = "flatten"
    RESHAPE = "reshape"
    CONCAT = "concat"
    ADD = "add"
    MULTIPLY = "multiply"
    SPLIT = "split"
    ATTENTION = "attention"
    MULTI_HEAD_ATTENTION = "multiHeadAttention"
    SELF_ATTENTION = "selfAttention"
    CROSS_ATTENTION = "crossAttention"
    LSTM = "lstm"
    GRU = "gru"
    RNN = "rnn"
    TRANSFORMER = "transformer"
    ENCODER_BLOCK = "encoderBlock"
    DECODER_BLOCK = "decoderBlock"
    RESIDUAL_BLOCK = "residualBlock"
    UPSAMPLE = "upsample"
    INTERPOLATE = "interpolate"
    PAD = "pad"
    CUSTOM = "custom"


@dataclass
class Position3D:
    """3D position"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class LayerParams:
    """Layer parameters"""
    in_channels: Optional[int] = None
    out_channels: Optional[int] = None
    in_features: Optional[int] = None
    out_features: Optional[int] = None
    kernel_size: Optional[Union[int, List[int]]] = None
    stride: Optional[Union[int, List[int]]] = None
    padding: Optional[Union[int, str, List[int]]] = None
    dilation: Optional[Union[int, List[int]]] = None
    groups: Optional[int] = None
    bias: Optional[bool] = None
    num_heads: Optional[int] = None
    hidden_size: Optional[int] = None
    dropout_rate: Optional[float] = None
    eps: Optional[float] = None
    momentum: Optional[float] = None
    affine: Optional[bool] = None
    num_embeddings: Optional[int] = None
    embedding_dim: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict with camelCase keys, excluding None values"""
        key_mapping = {
            'in_channels': 'inChannels',
            'out_channels': 'outChannels',
            'in_features': 'inFeatures',
            'out_features': 'outFeatures',
            'kernel_size': 'kernelSize',
            'num_heads': 'numHeads',
            'hidden_size': 'hiddenSize',
            'dropout_rate': 'dropoutRate',
            'num_embeddings': 'numEmbeddings',
            'embedding_dim': 'embeddingDim',
        }
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                camel_key = key_mapping.get(key, key)
                result[camel_key] = value
        return result


@dataclass
class NN3DNode:
    """Graph node representing a layer"""
    id: str
    type: str  # LayerType value
    name: str
    params: Optional[LayerParams] = None
    input_shape: Optional[List[Union[int, str]]] = None
    output_shape: Optional[List[Union[int, str]]] = None
    position: Optional[Position3D] = None
    attributes: Optional[Dict[str, Any]] = None
    group: Optional[str] = None
    depth: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'id': self.id,
            'type': self.type,
            'name': self.name,
        }
        if self.params:
            result['params'] = self.params.to_dict()
        if self.input_shape:
            result['inputShape'] = self.input_shape
        if self.output_shape:
            result['outputShape'] = self.output_shape
        if self.position:
            result['position'] = asdict(self.position)
        if self.attributes:
            result['attributes'] = self.attributes
        if self.group:
            result['group'] = self.group
        if self.depth is not None:
            result['depth'] = self.depth
        return result


@dataclass
class NN3DEdge:
    """Graph edge representing a connection"""
    source: str
    target: str
    id: Optional[str] = None
    source_port: Optional[int] = None
    target_port: Optional[int] = None
    tensor_shape: Optional[List[Union[int, str]]] = None
    dtype: Optional[str] = None
    label: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'source': self.source,
            'target': self.target,
        }
        if self.id:
            result['id'] = self.id
        if self.source_port is not None:
            result['sourcePort'] = self.source_port
        if self.target_port is not None:
            result['targetPort'] = self.target_port
        if self.tensor_shape:
            result['tensorShape'] = self.tensor_shape
        if self.dtype:
            result['dtype'] = self.dtype
        if self.label:
            result['label'] = self.label
        return result


@dataclass
class NN3DSubgraph:
    """Subgraph for grouping layers"""
    id: str
    name: str
    nodes: List[str] = field(default_factory=list)
    type: Optional[str] = None
    color: Optional[str] = None
    collapsed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'id': self.id,
            'name': self.name,
            'nodes': self.nodes,
        }
        if self.type:
            result['type'] = self.type
        if self.color:
            result['color'] = self.color
        if self.collapsed:
            result['collapsed'] = self.collapsed
        return result


@dataclass
class NN3DGraph:
    """Graph containing nodes and edges"""
    nodes: List[NN3DNode] = field(default_factory=list)
    edges: List[NN3DEdge] = field(default_factory=list)
    subgraphs: Optional[List[NN3DSubgraph]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'nodes': [n.to_dict() for n in self.nodes],
            'edges': [e.to_dict() for e in self.edges],
        }
        if self.subgraphs:
            result['subgraphs'] = [s.to_dict() for s in self.subgraphs]
        return result


@dataclass
class NN3DMetadata:
    """Model metadata"""
    name: str
    description: Optional[str] = None
    framework: Optional[str] = None
    author: Optional[str] = None
    created: Optional[str] = None
    tags: Optional[List[str]] = None
    input_shape: Optional[List[Union[int, str]]] = None
    output_shape: Optional[List[Union[int, str]]] = None
    total_params: Optional[int] = None
    trainable_params: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {'name': self.name}
        if self.description:
            result['description'] = self.description
        if self.framework:
            result['framework'] = self.framework
        if self.author:
            result['author'] = self.author
        if self.created:
            result['created'] = self.created
        if self.tags:
            result['tags'] = self.tags
        if self.input_shape:
            result['inputShape'] = self.input_shape
        if self.output_shape:
            result['outputShape'] = self.output_shape
        if self.total_params is not None:
            result['totalParams'] = self.total_params
        if self.trainable_params is not None:
            result['trainableParams'] = self.trainable_params
        return result


@dataclass
class VisualizationConfig:
    """Visualization configuration"""
    layout: str = "layered"
    theme: str = "dark"
    layer_spacing: float = 3.0
    node_scale: float = 1.0
    show_labels: bool = True
    show_edges: bool = True
    edge_style: str = "tube"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'layout': self.layout,
            'theme': self.theme,
            'layerSpacing': self.layer_spacing,
            'nodeScale': self.node_scale,
            'showLabels': self.show_labels,
            'showEdges': self.show_edges,
            'edgeStyle': self.edge_style,
        }


@dataclass
class NN3DModel:
    """Complete NN3D model"""
    metadata: NN3DMetadata
    graph: NN3DGraph
    version: str = "1.0.0"
    visualization: Optional[VisualizationConfig] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'version': self.version,
            'metadata': self.metadata.to_dict(),
            'graph': self.graph.to_dict(),
        }
        if self.visualization:
            result['visualization'] = self.visualization.to_dict()
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, filepath: str) -> None:
        """Save to .nn3d file"""
        with open(filepath, 'w') as f:
            f.write(self.to_json())
