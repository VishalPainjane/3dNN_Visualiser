/**
 * NN3D Schema TypeScript Types
 * Auto-generated from nn3d.schema.json
 */

// Schema version
export const NN3D_SCHEMA_VERSION = '1.0.0';

// Layer type enumeration
export type LayerType =
  | 'input'
  | 'output'
  | 'conv1d'
  | 'conv2d'
  | 'conv3d'
  | 'convTranspose2d'
  | 'depthwiseConv2d'
  | 'separableConv2d'
  | 'linear'
  | 'dense'
  | 'embedding'
  | 'batchNorm1d'
  | 'batchNorm2d'
  | 'layerNorm'
  | 'groupNorm'
  | 'instanceNorm'
  | 'dropout'
  | 'relu'
  | 'leakyRelu'
  | 'gelu'
  | 'silu'
  | 'sigmoid'
  | 'tanh'
  | 'softmax'
  | 'maxPool1d'
  | 'maxPool2d'
  | 'avgPool2d'
  | 'globalAvgPool'
  | 'adaptiveAvgPool'
  | 'flatten'
  | 'reshape'
  | 'concat'
  | 'add'
  | 'multiply'
  | 'split'
  | 'attention'
  | 'multiHeadAttention'
  | 'selfAttention'
  | 'crossAttention'
  | 'lstm'
  | 'gru'
  | 'rnn'
  | 'transformer'
  | 'encoderBlock'
  | 'decoderBlock'
  | 'residualBlock'
  | 'upsample'
  | 'interpolate'
  | 'pad'
  | 'custom';

// Tensor shape (dimensions can be numbers or dynamic strings)
export type TensorShape = (number | string)[];

// 3D position
export interface Position3D {
  x: number;
  y: number;
  z: number;
}

// Weight reference for loading weights
export interface WeightRef {
  url?: string;
  offset?: number;
  size?: number;
  dtype?: 'float16' | 'float32' | 'float64' | 'int32' | 'int64' | 'bool';
  shape?: TensorShape;
}

// Layer parameters
export interface LayerParams {
  inChannels?: number;
  outChannels?: number;
  inFeatures?: number;
  outFeatures?: number;
  kernelSize?: number | number[];
  stride?: number | number[];
  padding?: number | string | number[];
  dilation?: number | number[];
  groups?: number;
  bias?: boolean;
  numHeads?: number;
  hiddenSize?: number;
  dropoutRate?: number;
  eps?: number;
  momentum?: number;
  affine?: boolean;
  numEmbeddings?: number;
  embeddingDim?: number;
  axis?: number;
  scaleFactor?: number;
  mode?: string;
  [key: string]: unknown;
}

// Graph node (layer)
export interface NN3DNode {
  id: string;
  type: LayerType;
  name: string;
  params?: LayerParams;
  inputShape?: TensorShape;
  outputShape?: TensorShape;
  position?: Position3D;
  weights?: WeightRef;
  attributes?: Record<string, unknown>;
  group?: string;
  depth?: number;
}

// Graph edge (connection)
export interface NN3DEdge {
  id?: string;
  source: string;
  target: string;
  sourcePort?: number;
  targetPort?: number;
  tensorShape?: TensorShape;
  dtype?: 'float16' | 'float32' | 'float64' | 'int32' | 'int64' | 'bool';
  label?: string;
}

// Subgraph for grouping layers
export interface NN3DSubgraph {
  id: string;
  name: string;
  type?: 'sequential' | 'residual' | 'parallel' | 'attention' | 'custom';
  nodes: string[];
  color?: string;
  collapsed?: boolean;
}

// Graph structure
export interface NN3DGraph {
  nodes: NN3DNode[];
  edges: NN3DEdge[];
  subgraphs?: NN3DSubgraph[];
}

// Model metadata
export interface NN3DMetadata {
  name: string;
  description?: string;
  framework?: 'pytorch' | 'tensorflow' | 'keras' | 'onnx' | 'jax' | 'custom';
  author?: string;
  created?: string;
  tags?: string[];
  inputShape?: TensorShape;
  outputShape?: TensorShape;
  totalParams?: number;
  trainableParams?: number;
}

// Visualization configuration
export interface VisualizationConfig {
  layout?: 'layered' | 'force' | 'circular' | 'hierarchical' | 'custom';
  theme?: 'light' | 'dark' | 'blueprint';
  layerSpacing?: number;
  nodeScale?: number;
  colorScheme?: Record<string, string>;
  camera?: {
    position?: Position3D;
    target?: Position3D;
    fov?: number;
  };
  showLabels?: boolean;
  showEdges?: boolean;
  edgeStyle?: 'line' | 'tube' | 'arrow' | 'bezier';
}

// Activation data for visualization
export interface ActivationData {
  source?: 'file' | 'live' | 'embedded';
  url?: string;
  nodeActivations?: Record<string, {
    min?: number;
    max?: number;
    mean?: number;
    std?: number;
    histogram?: number[];
  }>;
}

// Complete NN3D model
export interface NN3DModel {
  version: string;
  metadata: NN3DMetadata;
  graph: NN3DGraph;
  visualization?: VisualizationConfig;
  activations?: ActivationData;
}

// Layer category for visualization grouping
export type LayerCategory =
  | 'input'
  | 'output'
  | 'convolution'
  | 'linear'
  | 'normalization'
  | 'activation'
  | 'pooling'
  | 'attention'
  | 'recurrent'
  | 'transform'
  | 'merge'
  | 'other';

// Map layer types to categories
export const LAYER_CATEGORIES: Record<LayerType, LayerCategory> = {
  input: 'input',
  output: 'output',
  conv1d: 'convolution',
  conv2d: 'convolution',
  conv3d: 'convolution',
  convTranspose2d: 'convolution',
  depthwiseConv2d: 'convolution',
  separableConv2d: 'convolution',
  linear: 'linear',
  dense: 'linear',
  embedding: 'linear',
  batchNorm1d: 'normalization',
  batchNorm2d: 'normalization',
  layerNorm: 'normalization',
  groupNorm: 'normalization',
  instanceNorm: 'normalization',
  dropout: 'normalization',
  relu: 'activation',
  leakyRelu: 'activation',
  gelu: 'activation',
  silu: 'activation',
  sigmoid: 'activation',
  tanh: 'activation',
  softmax: 'activation',
  maxPool1d: 'pooling',
  maxPool2d: 'pooling',
  avgPool2d: 'pooling',
  globalAvgPool: 'pooling',
  adaptiveAvgPool: 'pooling',
  flatten: 'transform',
  reshape: 'transform',
  concat: 'merge',
  add: 'merge',
  multiply: 'merge',
  split: 'merge',
  attention: 'attention',
  multiHeadAttention: 'attention',
  selfAttention: 'attention',
  crossAttention: 'attention',
  lstm: 'recurrent',
  gru: 'recurrent',
  rnn: 'recurrent',
  transformer: 'attention',
  encoderBlock: 'attention',
  decoderBlock: 'attention',
  residualBlock: 'merge',
  upsample: 'transform',
  interpolate: 'transform',
  pad: 'transform',
  custom: 'other',
};

// Default colors for layer categories
export const DEFAULT_CATEGORY_COLORS: Record<LayerCategory, string> = {
  input: '#4CAF50',
  output: '#F44336',
  convolution: '#2196F3',
  linear: '#9C27B0',
  normalization: '#FF9800',
  activation: '#FFEB3B',
  pooling: '#00BCD4',
  attention: '#E91E63',
  recurrent: '#673AB7',
  transform: '#795548',
  merge: '#607D8B',
  other: '#9E9E9E',
};
