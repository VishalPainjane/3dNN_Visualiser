/**
 * SafeTensors parser
 * Parses SafeTensors format header to extract tensor metadata
 * SafeTensors is popular for LLM weights storage
 */

import type { NN3DModel, LayerType } from '@/schema/types';
import type { ParseResult, FormatParser, ExtractedLayer } from './types';
import { detectFormatFromExtension } from './format-detector';

/**
 * SafeTensors header structure
 */
interface SafeTensorsHeader {
  [key: string]: {
    dtype: string;
    shape: number[];
    data_offsets: [number, number];
  } | {
    __metadata__?: Record<string, string>;
  };
}

/**
 * Try to infer layer type from tensor name
 */
function inferLayerType(tensorName: string): LayerType {
  const name = tensorName.toLowerCase();
  
  // Attention patterns
  if (name.includes('attn') || name.includes('attention')) {
    if (name.includes('q_proj') || name.includes('query')) return 'multiHeadAttention';
    if (name.includes('k_proj') || name.includes('key')) return 'multiHeadAttention';
    if (name.includes('v_proj') || name.includes('value')) return 'multiHeadAttention';
    if (name.includes('o_proj') || name.includes('out')) return 'multiHeadAttention';
    return 'multiHeadAttention';
  }
  
  // Normalization
  if (name.includes('layernorm') || name.includes('layer_norm') || name.includes('ln_')) {
    return 'layerNorm';
  }
  if (name.includes('batchnorm') || name.includes('batch_norm') || name.includes('bn_')) {
    return 'batchNorm2d';
  }
  if (name.includes('groupnorm') || name.includes('group_norm')) {
    return 'groupNorm';
  }
  
  // Linear/MLP
  if (name.includes('mlp') || name.includes('ffn') || name.includes('fc') || 
      name.includes('linear') || name.includes('dense') || name.includes('proj')) {
    return 'linear';
  }
  
  // Convolution
  if (name.includes('conv')) {
    if (name.includes('conv1d')) return 'conv1d';
    if (name.includes('conv3d')) return 'conv3d';
    return 'conv2d';
  }
  
  // Embedding
  if (name.includes('embed') || name.includes('wte') || name.includes('wpe')) {
    return 'embedding';
  }
  
  // Output head
  if (name.includes('lm_head') || name.includes('classifier') || name.includes('head')) {
    return 'linear';
  }
  
  return 'linear'; // Default to linear for weight tensors
}

/**
 * Group tensors by layer
 */
function groupTensorsByLayer(header: SafeTensorsHeader): Map<string, ExtractedLayer> {
  const layers = new Map<string, ExtractedLayer>();
  
  for (const [name, info] of Object.entries(header)) {
    if (name === '__metadata__' || !('dtype' in info)) continue;
    
    // Extract layer name (remove .weight, .bias, etc.)
    let layerName = name
      .replace(/\.(weight|bias|gamma|beta|running_mean|running_var)$/i, '')
      .replace(/\.(q_proj|k_proj|v_proj|o_proj|out_proj)$/i, '')
      .replace(/\.0$/i, ''); // Remove trailing .0
    
    // Get or create layer
    if (!layers.has(layerName)) {
      layers.set(layerName, {
        id: layerName.replace(/\./g, '_'),
        name: layerName,
        type: inferLayerType(layerName),
        params: {},
        outputShape: undefined,
      });
    }
    
    const layer = layers.get(layerName)!;
    
    // Infer shape info from tensor
    if (name.endsWith('.weight') || name.endsWith('_weight')) {
      const shape = info.shape;
      if (shape.length === 2) {
        // Linear layer: [out_features, in_features]
        layer.params = {
          ...layer.params,
          inFeatures: shape[1],
          outFeatures: shape[0],
        };
        layer.inputShape = [1, shape[1]];
        layer.outputShape = [1, shape[0]];
      } else if (shape.length === 4) {
        // Conv layer: [out_channels, in_channels, H, W]
        layer.params = {
          ...layer.params,
          outChannels: shape[0],
          inChannels: shape[1],
          kernelSize: [shape[2], shape[3]],
        };
        layer.type = 'conv2d';
      } else if (shape.length === 1) {
        // Embedding or norm: [hidden_size]
        if (layer.type === 'embedding') {
          layer.params = {
            ...layer.params,
            embeddingDim: shape[0],
          };
        }
      }
    }
    
    // Store tensor info
    if (!layer.attributes) layer.attributes = {};
    (layer.attributes as any)[`tensor_${name.split('.').pop()}`] = {
      dtype: info.dtype,
      shape: info.shape,
    };
  }
  
  return layers;
}

/**
 * SafeTensors format parser
 */
export const SafeTensorsParser: FormatParser = {
  extensions: ['.safetensors'],
  
  async canParse(file: File): Promise<boolean> {
    if (!file.name.toLowerCase().endsWith('.safetensors')) return false;
    
    // Verify by checking header structure
    try {
      const headerLen = await readHeaderLength(file);
      return headerLen > 0 && headerLen < 10_000_000; // Reasonable header size
    } catch {
      return false;
    }
  },
  
  async parse(file: File): Promise<ParseResult> {
    const format = detectFormatFromExtension(file.name);
    const warnings: string[] = [];
    
    try {
      // Read header length (first 8 bytes, little-endian u64)
      const headerLen = await readHeaderLength(file);
      
      if (headerLen > 100_000_000) {
        throw new Error('Header too large');
      }
      
      // Read header JSON
      const headerBytes = await file.slice(8, 8 + Number(headerLen)).text();
      const header: SafeTensorsHeader = JSON.parse(headerBytes);
      
      // Extract metadata
      const metadata = (header.__metadata__ as Record<string, string>) || {};
      delete header.__metadata__;
      
      // Group tensors by layer
      const layerMap = groupTensorsByLayer(header);
      const layers = Array.from(layerMap.values());
      
      if (layers.length === 0) {
        throw new Error('No layers found in SafeTensors file');
      }
      
      // Sort layers by name to maintain some order
      layers.sort((a, b) => a.name.localeCompare(b.name));
      
      // Try to infer connections based on naming patterns
      const connections = inferConnectionsFromNames(layers);
      
      // Calculate total parameters
      let totalParams = 0;
      for (const [, info] of Object.entries(header)) {
        if ('shape' in info) {
          totalParams += info.shape.reduce((a, b) => a * b, 1);
        }
      }
      
      warnings.push('Layer connections inferred from naming patterns. Structure may not be exact.');
      
      // Build model
      const model: NN3DModel = {
        version: '1.0.0',
        metadata: {
          name: file.name.replace('.safetensors', ''),
          description: `Imported from SafeTensors (${layers.length} layers, ${formatNumber(totalParams)} parameters)`,
          framework: (metadata.format as 'pytorch' | 'tensorflow' | 'keras' | 'onnx' | 'jax' | 'custom') || 'pytorch',
          created: new Date().toISOString(),
          tags: ['safetensors', 'imported', 'llm'],
          totalParams,
          trainableParams: totalParams,
        },
        graph: {
          nodes: [
            // Add input node
            {
              id: 'input',
              type: 'input',
              name: 'Input',
              depth: 0,
            },
            // Add all layers
            ...layers.map((layer, i) => ({
              id: layer.id,
              type: layer.type as LayerType,
              name: layer.name,
              params: layer.params as any,
              inputShape: layer.inputShape,
              outputShape: layer.outputShape,
              depth: i + 1,
            })),
            // Add output node
            {
              id: 'output',
              type: 'output',
              name: 'Output',
              depth: layers.length + 1,
            },
          ],
          edges: [
            // Input to first layer
            { source: 'input', target: layers[0]?.id || 'output' },
            // Layer connections
            ...connections,
            // Last layer to output
            { source: layers[layers.length - 1]?.id || 'input', target: 'output' },
          ],
        },
        visualization: {
          layout: 'layered',
          theme: 'dark',
          layerSpacing: 2.0,
          nodeScale: 0.8,
          showLabels: true,
          showEdges: true,
          edgeStyle: 'bezier',
        },
      };
      
      return {
        success: true,
        model,
        warnings,
        format,
        inferredStructure: true,
      };
      
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to parse SafeTensors file',
        warnings,
        format,
        inferredStructure: false,
      };
    }
  }
};

/**
 * Read the header length from SafeTensors file
 */
async function readHeaderLength(file: File): Promise<number> {
  const buffer = await file.slice(0, 8).arrayBuffer();
  const view = new DataView(buffer);
  // Little-endian u64, but we only need the lower 32 bits for reasonable headers
  return view.getUint32(0, true);
}

/**
 * Infer connections from layer naming patterns
 */
function inferConnectionsFromNames(layers: ExtractedLayer[]): Array<{ source: string; target: string }> {
  const connections: Array<{ source: string; target: string }> = [];
  
  // Group layers by block/module
  const blockPattern = /^(.*?)\.(\d+)\.(.*)$/;
  const blocks = new Map<string, ExtractedLayer[]>();
  const standalone: ExtractedLayer[] = [];
  
  for (const layer of layers) {
    const match = layer.name.match(blockPattern);
    if (match) {
      const blockKey = `${match[1]}.${match[2]}`;
      if (!blocks.has(blockKey)) {
        blocks.set(blockKey, []);
      }
      blocks.get(blockKey)!.push(layer);
    } else {
      standalone.push(layer);
    }
  }
  
  // Connect sequential layers
  const orderedLayers = [...layers];
  
  for (let i = 0; i < orderedLayers.length - 1; i++) {
    connections.push({
      source: orderedLayers[i].id,
      target: orderedLayers[i + 1].id,
    });
  }
  
  return connections;
}

/**
 * Format large numbers for display
 */
function formatNumber(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toString();
}

export default SafeTensorsParser;
