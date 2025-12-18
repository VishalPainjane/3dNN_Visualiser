/**
 * ONNX model parser
 * Parses ONNX protobuf files directly in the browser
 */

import type { NN3DModel, LayerType } from '@/schema/types';
import type { ParseResult, FormatParser, ExtractedLayer, ExtractedConnection } from './types';
import { detectFormatFromExtension } from './format-detector';
import * as ort from 'onnxruntime-web';

/**
 * Map ONNX operator types to NN3D layer types
 */
const ONNX_OP_MAP: Record<string, LayerType> = {
  // Convolution
  'Conv': 'conv2d',
  'ConvTranspose': 'convTranspose2d',
  'DepthwiseConv': 'depthwiseConv2d',
  
  // Linear
  'Gemm': 'linear',
  'MatMul': 'linear',
  'MatMulInteger': 'linear',
  
  // Activation
  'Relu': 'relu',
  'LeakyRelu': 'leakyRelu',
  'Sigmoid': 'sigmoid',
  'Tanh': 'tanh',
  'Softmax': 'softmax',
  'LogSoftmax': 'softmax',
  'Gelu': 'gelu',
  'Silu': 'silu',
  'HardSigmoid': 'sigmoid',
  'Mish': 'gelu',
  'Elu': 'relu',
  'Selu': 'relu',
  'Celu': 'relu',
  'Softplus': 'relu',
  
  // Normalization
  'BatchNormalization': 'batchNorm2d',
  'InstanceNormalization': 'instanceNorm',
  'LayerNormalization': 'layerNorm',
  'GroupNormalization': 'groupNorm',
  'LpNormalization': 'layerNorm',
  
  // Pooling
  'MaxPool': 'maxPool2d',
  'AveragePool': 'avgPool2d',
  'GlobalAveragePool': 'globalAvgPool',
  'GlobalMaxPool': 'maxPool2d',
  'AdaptiveAvgPool2d': 'adaptiveAvgPool',
  'LpPool': 'avgPool2d',
  
  // Attention
  'Attention': 'multiHeadAttention',
  'MultiHeadAttention': 'multiHeadAttention',
  
  // Recurrent
  'LSTM': 'lstm',
  'GRU': 'gru',
  'RNN': 'rnn',
  
  // Operations
  'Add': 'add',
  'Sum': 'add',
  'Sub': 'add',
  'Mul': 'multiply',
  'Div': 'multiply',
  'Concat': 'concat',
  'Split': 'split',
  'Reshape': 'reshape',
  'Flatten': 'flatten',
  'Squeeze': 'reshape',
  'Unsqueeze': 'reshape',
  'Transpose': 'reshape',
  'Slice': 'reshape',
  'Gather': 'embedding',
  'Scatter': 'reshape',
  'Pad': 'pad',
  
  // Embedding
  'Embedding': 'embedding',
  
  // Regularization
  'Dropout': 'dropout',
  
  // Reduction
  'ReduceMean': 'globalAvgPool',
  'ReduceSum': 'add',
  'ReduceMax': 'maxPool2d',
  'ReduceMin': 'maxPool2d',
  
  // Element-wise
  'Clip': 'relu',
  'Abs': 'relu',
  'Neg': 'multiply',
  'Exp': 'relu',
  'Log': 'relu',
  'Sqrt': 'relu',
  'Pow': 'multiply',
  
  // Misc
  'Shape': 'reshape',
  'Size': 'reshape',
  'Resize': 'upsample',
  'Upsample': 'upsample',
};



/**
 * ONNX format parser
 */
export const OnnxParser: FormatParser = {
  extensions: ['.onnx'],
  
  async canParse(file: File): Promise<boolean> {
    return file.name.toLowerCase().endsWith('.onnx');
  },
  
  async parse(file: File): Promise<ParseResult> {
    const format = detectFormatFromExtension(file.name);
    const warnings: string[] = [];
    
    try {
      const buffer = await file.arrayBuffer();
      
      // Use ONNX Runtime to parse the model
      const session = await ort.InferenceSession.create(buffer, {
        executionProviders: ['wasm'],
      });
      
      // Extract model info from session
      const layers: ExtractedLayer[] = [];
      const inputNames = [...session.inputNames];
      const outputNames = [...session.outputNames];
      
      // Create input nodes
      for (const inputName of inputNames) {
        layers.push({
          id: inputName,
          name: inputName,
          type: 'input',
          outputShape: undefined, // Will be set if available
        });
      }
      
      // Since onnxruntime-web doesn't expose the full graph,
      // we need to parse the protobuf directly for detailed structure
      // For now, we'll create a simplified representation
      
      // Try to parse with protobuf for full graph
      const model = await parseOnnxProtobuf(buffer, warnings);
      
      if (model) {
        return {
          success: true,
          model,
          warnings,
          format,
          inferredStructure: false,
        };
      }
      
      // Fallback: create minimal model from session info
      const fallbackModel = createFallbackModel(
        file.name.replace('.onnx', ''),
        inputNames,
        outputNames,
      );
      
      warnings.push('Full graph structure not available. Showing simplified input/output view.');
      
      return {
        success: true,
        model: fallbackModel,
        warnings,
        format,
        inferredStructure: true,
      };
      
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to parse ONNX model',
        warnings,
        format,
        inferredStructure: false,
      };
    }
  }
};

/**
 * Parse ONNX protobuf directly for full graph structure
 */
async function parseOnnxProtobuf(buffer: ArrayBuffer, warnings: string[]): Promise<NN3DModel | null> {
  try {
    // Minimal ONNX protobuf parser
    // ONNX uses Protocol Buffers, we'll parse the essential parts
    const data = new Uint8Array(buffer);
    
    // ONNX ModelProto structure (simplified):
    // - field 7: graph (GraphProto)
    //   - field 1: node (repeated NodeProto)
    //   - field 11: input (repeated ValueInfoProto)
    //   - field 12: output (repeated ValueInfoProto)
    
    const model = decodeOnnxModel(data);
    if (!model || !model.graph) {
      warnings.push('Could not decode ONNX protobuf structure');
      return null;
    }
    
    const graph = model.graph;
    const nodes: ExtractedLayer[] = [];
    const edges: ExtractedConnection[] = [];
    const tensorShapes = new Map<string, number[]>();
    
    // Extract value info shapes
    for (const info of [...(graph.input || []), ...(graph.output || []), ...(graph.valueInfo || [])]) {
      if (info.name && info.type?.tensorType?.shape?.dim) {
        const shape = info.type.tensorType.shape.dim.map((d: any) => 
          d.dimValue ? Number(d.dimValue) : -1
        );
        tensorShapes.set(info.name, shape);
      }
    }
    
    // Add inputs
    for (const input of graph.input || []) {
      // Skip initializers (weights)
      const isInitializer = (graph.initializer || []).some((init: any) => init.name === input.name);
      if (isInitializer) continue;
      
      nodes.push({
        id: input.name,
        name: input.name,
        type: 'input',
        outputShape: tensorShapes.get(input.name),
      });
    }
    
    // Track output name to node ID mapping
    const outputToNode = new Map<string, string>();
    for (const node of nodes) {
      outputToNode.set(node.id, node.id);
    }
    
    // Process nodes
    for (let i = 0; i < (graph.node || []).length; i++) {
      const node = graph.node[i];
      const nodeId = node.name || `node_${i}`;
      const opType = node.opType || 'Unknown';
      
      // Map to NN3D layer type
      const layerType = ONNX_OP_MAP[opType] || 'custom';
      
      // Get output shape from first output
      const outputName = node.output?.[0];
      const outputShape = outputName ? tensorShapes.get(outputName) : undefined;
      
      nodes.push({
        id: nodeId,
        name: node.name || opType,
        type: layerType,
        outputShape: outputShape ? [...outputShape] : undefined,
        attributes: { onnxOpType: opType },
      });
      
      // Map outputs to this node
      for (const output of node.output || []) {
        outputToNode.set(output, nodeId);
      }
      
      // Create edges from inputs
      for (const input of node.input || []) {
        const sourceNode = outputToNode.get(input);
        if (sourceNode) {
          edges.push({
            source: sourceNode,
            target: nodeId,
          });
        }
      }
    }
    
    // Add outputs
    for (const output of graph.output || []) {
      const outputId = `output_${output.name}`;
      nodes.push({
        id: outputId,
        name: output.name,
        type: 'output',
        inputShape: tensorShapes.get(output.name),
      });
      
      const sourceNode = outputToNode.get(output.name);
      if (sourceNode) {
        edges.push({
          source: sourceNode,
          target: outputId,
        });
      }
    }
    
    // Convert to NN3D model
    return convertToNN3D(
      graph.name || 'ONNX Model',
      nodes,
      edges,
      {
        framework: 'onnx',
        opsetVersion: model.opsetImport?.[0]?.version,
        producerName: model.producerName,
        producerVersion: model.producerVersion,
      }
    );
    
  } catch (error) {
    warnings.push(`Protobuf parsing error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    return null;
  }
}

/**
 * Minimal ONNX protobuf decoder
 */
function decodeOnnxModel(data: Uint8Array): any {
  const reader = new ProtobufReader(data);
  const model: any = {};
  
  while (reader.pos < data.length) {
    const tag = reader.readVarint();
    const fieldNum = tag >> 3;
    const wireType = tag & 0x7;
    
    switch (fieldNum) {
      case 1: // ir_version
        model.irVersion = reader.readVarint();
        break;
      case 2: // opset_import
        model.opsetImport = model.opsetImport || [];
        model.opsetImport.push(reader.readMessage(decodeOpsetImport));
        break;
      case 3: // producer_name
        model.producerName = reader.readString();
        break;
      case 4: // producer_version
        model.producerVersion = reader.readString();
        break;
      case 7: // graph
        model.graph = reader.readMessage(decodeGraph);
        break;
      default:
        reader.skipField(wireType);
    }
  }
  
  return model;
}

function decodeOpsetImport(data: Uint8Array): any {
  const reader = new ProtobufReader(data);
  const opset: any = {};
  
  while (reader.pos < data.length) {
    const tag = reader.readVarint();
    const fieldNum = tag >> 3;
    const wireType = tag & 0x7;
    
    switch (fieldNum) {
      case 1: // domain
        opset.domain = reader.readString();
        break;
      case 2: // version
        opset.version = reader.readVarint();
        break;
      default:
        reader.skipField(wireType);
    }
  }
  
  return opset;
}

function decodeGraph(data: Uint8Array): any {
  const reader = new ProtobufReader(data);
  const graph: any = { node: [], input: [], output: [], initializer: [], valueInfo: [] };
  
  while (reader.pos < data.length) {
    const tag = reader.readVarint();
    const fieldNum = tag >> 3;
    const wireType = tag & 0x7;
    
    switch (fieldNum) {
      case 1: // node
        graph.node.push(reader.readMessage(decodeNode));
        break;
      case 2: // name
        graph.name = reader.readString();
        break;
      case 5: // initializer
        graph.initializer.push(reader.readMessage(decodeTensor));
        break;
      case 11: // input
        graph.input.push(reader.readMessage(decodeValueInfo));
        break;
      case 12: // output
        graph.output.push(reader.readMessage(decodeValueInfo));
        break;
      case 13: // value_info
        graph.valueInfo.push(reader.readMessage(decodeValueInfo));
        break;
      default:
        reader.skipField(wireType);
    }
  }
  
  return graph;
}

function decodeNode(data: Uint8Array): any {
  const reader = new ProtobufReader(data);
  const node: any = { input: [], output: [], attribute: [] };
  
  while (reader.pos < data.length) {
    const tag = reader.readVarint();
    const fieldNum = tag >> 3;
    const wireType = tag & 0x7;
    
    switch (fieldNum) {
      case 1: // input
        node.input.push(reader.readString());
        break;
      case 2: // output
        node.output.push(reader.readString());
        break;
      case 3: // name
        node.name = reader.readString();
        break;
      case 4: // op_type
        node.opType = reader.readString();
        break;
      case 5: // attribute
        node.attribute.push(reader.readMessage(decodeAttribute));
        break;
      default:
        reader.skipField(wireType);
    }
  }
  
  return node;
}

function decodeAttribute(data: Uint8Array): any {
  const reader = new ProtobufReader(data);
  const attr: any = {};
  
  while (reader.pos < data.length) {
    const tag = reader.readVarint();
    const fieldNum = tag >> 3;
    const wireType = tag & 0x7;
    
    switch (fieldNum) {
      case 1: // name
        attr.name = reader.readString();
        break;
      case 2: // f (float)
        attr.f = reader.readFloat();
        break;
      case 3: // i (int)
        attr.i = reader.readVarint();
        break;
      case 4: // s (bytes/string)
        attr.s = reader.readString();
        break;
      case 7: // ints
        attr.ints = attr.ints || [];
        attr.ints.push(reader.readVarint());
        break;
      default:
        reader.skipField(wireType);
    }
  }
  
  return attr;
}

function decodeValueInfo(data: Uint8Array): any {
  const reader = new ProtobufReader(data);
  const info: any = {};
  
  while (reader.pos < data.length) {
    const tag = reader.readVarint();
    const fieldNum = tag >> 3;
    const wireType = tag & 0x7;
    
    switch (fieldNum) {
      case 1: // name
        info.name = reader.readString();
        break;
      case 2: // type
        info.type = reader.readMessage(decodeTypeProto);
        break;
      default:
        reader.skipField(wireType);
    }
  }
  
  return info;
}

function decodeTypeProto(data: Uint8Array): any {
  const reader = new ProtobufReader(data);
  const type: any = {};
  
  while (reader.pos < data.length) {
    const tag = reader.readVarint();
    const fieldNum = tag >> 3;
    const wireType = tag & 0x7;
    
    switch (fieldNum) {
      case 1: // tensor_type
        type.tensorType = reader.readMessage(decodeTensorType);
        break;
      default:
        reader.skipField(wireType);
    }
  }
  
  return type;
}

function decodeTensorType(data: Uint8Array): any {
  const reader = new ProtobufReader(data);
  const tensor: any = {};
  
  while (reader.pos < data.length) {
    const tag = reader.readVarint();
    const fieldNum = tag >> 3;
    const wireType = tag & 0x7;
    
    switch (fieldNum) {
      case 1: // elem_type
        tensor.elemType = reader.readVarint();
        break;
      case 2: // shape
        tensor.shape = reader.readMessage(decodeTensorShape);
        break;
      default:
        reader.skipField(wireType);
    }
  }
  
  return tensor;
}

function decodeTensorShape(data: Uint8Array): any {
  const reader = new ProtobufReader(data);
  const shape: any = { dim: [] };
  
  while (reader.pos < data.length) {
    const tag = reader.readVarint();
    const fieldNum = tag >> 3;
    const wireType = tag & 0x7;
    
    switch (fieldNum) {
      case 1: // dim
        shape.dim.push(reader.readMessage(decodeDimension));
        break;
      default:
        reader.skipField(wireType);
    }
  }
  
  return shape;
}

function decodeDimension(data: Uint8Array): any {
  const reader = new ProtobufReader(data);
  const dim: any = {};
  
  while (reader.pos < data.length) {
    const tag = reader.readVarint();
    const fieldNum = tag >> 3;
    const wireType = tag & 0x7;
    
    switch (fieldNum) {
      case 1: // dim_value
        dim.dimValue = reader.readVarint();
        break;
      case 2: // dim_param
        dim.dimParam = reader.readString();
        break;
      default:
        reader.skipField(wireType);
    }
  }
  
  return dim;
}

function decodeTensor(data: Uint8Array): any {
  const reader = new ProtobufReader(data);
  const tensor: any = {};
  
  while (reader.pos < data.length) {
    const tag = reader.readVarint();
    const fieldNum = tag >> 3;
    const wireType = tag & 0x7;
    
    switch (fieldNum) {
      case 1: // dims
        tensor.dims = tensor.dims || [];
        tensor.dims.push(reader.readVarint());
        break;
      case 2: // data_type
        tensor.dataType = reader.readVarint();
        break;
      case 8: // name
        tensor.name = reader.readString();
        break;
      default:
        reader.skipField(wireType);
    }
  }
  
  return tensor;
}

/**
 * Minimal protobuf reader
 */
class ProtobufReader {
  pos = 0;
  
  constructor(private data: Uint8Array) {}
  
  readVarint(): number {
    let result = 0;
    let shift = 0;
    
    while (this.pos < this.data.length) {
      const byte = this.data[this.pos++];
      result |= (byte & 0x7f) << shift;
      if ((byte & 0x80) === 0) break;
      shift += 7;
    }
    
    return result >>> 0;
  }
  
  readString(): string {
    const len = this.readVarint();
    const bytes = this.data.slice(this.pos, this.pos + len);
    this.pos += len;
    return new TextDecoder().decode(bytes);
  }
  
  readBytes(): Uint8Array {
    const len = this.readVarint();
    const bytes = this.data.slice(this.pos, this.pos + len);
    this.pos += len;
    return bytes;
  }
  
  readFloat(): number {
    const view = new DataView(this.data.buffer, this.data.byteOffset + this.pos, 4);
    this.pos += 4;
    return view.getFloat32(0, true);
  }
  
  readMessage<T>(decoder: (data: Uint8Array) => T): T {
    const bytes = this.readBytes();
    return decoder(bytes);
  }
  
  skipField(wireType: number): void {
    switch (wireType) {
      case 0: // Varint
        this.readVarint();
        break;
      case 1: // 64-bit
        this.pos += 8;
        break;
      case 2: // Length-delimited
        const len = this.readVarint();
        this.pos += len;
        break;
      case 5: // 32-bit
        this.pos += 4;
        break;
    }
  }
}

/**
 * Convert extracted structure to NN3D model
 */
function convertToNN3D(
  name: string,
  layers: ExtractedLayer[],
  connections: ExtractedConnection[],
  metadata: Record<string, unknown> = {}
): NN3DModel {
  // Compute depth for each layer
  const depthMap = new Map<string, number>();
  const inDegree = new Map<string, number>();
  const adjacency = new Map<string, string[]>();
  
  // Initialize
  for (const layer of layers) {
    depthMap.set(layer.id, 0);
    inDegree.set(layer.id, 0);
    adjacency.set(layer.id, []);
  }
  
  // Build adjacency and in-degree
  for (const conn of connections) {
    const targets = adjacency.get(conn.source) || [];
    targets.push(conn.target);
    adjacency.set(conn.source, targets);
    inDegree.set(conn.target, (inDegree.get(conn.target) || 0) + 1);
  }
  
  // Topological sort to compute depths
  const queue = layers.filter(l => (inDegree.get(l.id) || 0) === 0).map(l => l.id);
  
  while (queue.length > 0) {
    const nodeId = queue.shift()!;
    const currentDepth = depthMap.get(nodeId) || 0;
    
    for (const targetId of adjacency.get(nodeId) || []) {
      depthMap.set(targetId, Math.max(depthMap.get(targetId) || 0, currentDepth + 1));
      const remaining = (inDegree.get(targetId) || 0) - 1;
      inDegree.set(targetId, remaining);
      if (remaining === 0) {
        queue.push(targetId);
      }
    }
  }
  
  // Count total parameters (rough estimate)
  let totalParams = 0;
  for (const layer of layers) {
    if (layer.type === 'linear' && layer.inputShape && layer.outputShape) {
      const inSize = layer.inputShape[layer.inputShape.length - 1] || 1;
      const outSize = layer.outputShape[layer.outputShape.length - 1] || 1;
      totalParams += inSize * outSize;
    }
  }
  
  return {
    version: '1.0.0',
    metadata: {
      name,
      description: `Imported from ONNX model`,
      framework: 'onnx',
      created: new Date().toISOString(),
      tags: ['onnx', 'imported'],
      totalParams,
      trainableParams: totalParams,
      ...metadata,
    },
    graph: {
      nodes: layers.map(layer => ({
        id: layer.id,
        type: layer.type as LayerType,
        name: layer.name,
        params: layer.params as any,
        inputShape: layer.inputShape,
        outputShape: layer.outputShape,
        depth: depthMap.get(layer.id) || 0,
      })),
      edges: connections.map(conn => ({
        source: conn.source,
        target: conn.target,
      })),
    },
    visualization: {
      layout: 'layered',
      theme: 'dark',
      layerSpacing: 2.5,
      nodeScale: 1.0,
      showLabels: true,
      showEdges: true,
      edgeStyle: 'bezier',
    },
  };
}

/**
 * Create fallback model when full parsing fails
 */
function createFallbackModel(
  name: string,
  inputNames: string[],
  outputNames: string[]
): NN3DModel {
  const nodes = [
    ...inputNames.map((name, i) => ({
      id: `input_${i}`,
      type: 'input' as LayerType,
      name,
      depth: 0,
    })),
    {
      id: 'model',
      type: 'custom' as LayerType,
      name: 'Model (structure unavailable)',
      depth: 1,
    },
    ...outputNames.map((name, i) => ({
      id: `output_${i}`,
      type: 'output' as LayerType,
      name,
      depth: 2,
    })),
  ];
  
  const edges = [
    ...inputNames.map((_, i) => ({
      source: `input_${i}`,
      target: 'model',
    })),
    ...outputNames.map((_, i) => ({
      source: 'model',
      target: `output_${i}`,
    })),
  ];
  
  return {
    version: '1.0.0',
    metadata: {
      name,
      framework: 'onnx',
      created: new Date().toISOString(),
    },
    graph: { nodes, edges },
    visualization: {
      layout: 'layered',
      theme: 'dark',
    },
  };
}

export default OnnxParser;
