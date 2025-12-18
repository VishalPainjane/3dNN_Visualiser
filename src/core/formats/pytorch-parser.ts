/**
 * PyTorch format parser
 * Parses PyTorch .pt, .pth, .ckpt, .bin files by reading the pickle structure
 * Inspired by Netron's approach to extracting layer information from state dicts
 */

import type { NN3DModel, LayerType } from '@/schema/types';
import type { ParseResult, FormatParser, ExtractedLayer } from './types';
import { detectFormatFromExtension } from './format-detector';
import * as pako from 'pako';

/**
 * Minimal pickle parser for reading PyTorch files
 * Handles the subset of pickle opcodes used by PyTorch
 */
class PickleReader {
  private pos = 0;
  private data: DataView;
  private bytes: Uint8Array;
  private memo: Map<number, unknown> = new Map();
  private stack: unknown[] = [];
  private metastack: unknown[][] = [];
  
  constructor(buffer: ArrayBuffer) {
    this.bytes = new Uint8Array(buffer);
    this.data = new DataView(buffer);
  }
  
  private readByte(): number {
    return this.bytes[this.pos++];
  }
  
  private readBytes(n: number): Uint8Array {
    const result = this.bytes.slice(this.pos, this.pos + n);
    this.pos += n;
    return result;
  }
  
  private readUint16(): number {
    const result = this.data.getUint16(this.pos, true);
    this.pos += 2;
    return result;
  }
  
  private readUint32(): number {
    const result = this.data.getUint32(this.pos, true);
    this.pos += 4;
    return result;
  }
  
  private readInt32(): number {
    const result = this.data.getInt32(this.pos, true);
    this.pos += 4;
    return result;
  }
  
  private readFloat64(): number {
    const result = this.data.getFloat64(this.pos, true);
    this.pos += 8;
    return result;
  }
  
  private readLine(): string {
    let result = '';
    while (this.pos < this.bytes.length) {
      const char = this.bytes[this.pos++];
      if (char === 0x0a) break;
      result += String.fromCharCode(char);
    }
    return result;
  }
  
  private readString(len: number): string {
    const bytes = this.readBytes(len);
    return new TextDecoder().decode(bytes);
  }
  
  private readShortBinString(): string {
    const len = this.readByte();
    return this.readString(len);
  }
  
  private readBinString(): string {
    const len = this.readUint32();
    return this.readString(len);
  }

  /**
   * Parse pickle stream and return the result
   */
  parse(): unknown {
    while (this.pos < this.bytes.length) {
      const opcode = this.readByte();
      
      switch (opcode) {
        case 0x80: // PROTO
          this.readByte();
          break;
        case 0x7d: // EMPTY_DICT
          this.stack.push(new Map<string, unknown>());
          break;
        case 0x5d: // EMPTY_LIST
          this.stack.push([]);
          break;
        case 0x29: // EMPTY_TUPLE
          this.stack.push([]);
          break;
        case 0x4e: // NONE
          this.stack.push(null);
          break;
        case 0x88: // NEWTRUE
          this.stack.push(true);
          break;
        case 0x89: // NEWFALSE
          this.stack.push(false);
          break;
        case 0x4a: // BININT
          this.stack.push(this.readInt32());
          break;
        case 0x4b: // BININT1
          this.stack.push(this.readByte());
          break;
        case 0x4d: // BININT2
          this.stack.push(this.readUint16());
          break;
        case 0x47: // BINFLOAT
          this.stack.push(this.readFloat64());
          break;
        case 0x8a: { // LONG1
          const n = this.readByte();
          if (n === 0) {
            this.stack.push(0);
          } else {
            const bytes = this.readBytes(n);
            let value = 0;
            for (let i = 0; i < n; i++) {
              value |= bytes[i] << (8 * i);
            }
            this.stack.push(value);
          }
          break;
        }
        case 0x55: // SHORT_BINSTRING
        case 0x8c: // SHORT_BINUNICODE
          this.stack.push(this.readShortBinString());
          break;
        case 0x58: // BINUNICODE
        case 0x54: // BINSTRING
          this.stack.push(this.readBinString());
          break;
        case 0x8d: { // BINUNICODE8
          const len8 = Number(this.data.getBigUint64(this.pos, true));
          this.pos += 8;
          this.stack.push(this.readString(len8));
          break;
        }
        case 0x28: // MARK
          this.metastack.push(this.stack);
          this.stack = [];
          break;
        case 0x74: { // TUPLE
          const tupleItems = this.stack;
          this.stack = this.metastack.pop() || [];
          this.stack.push(tupleItems);
          break;
        }
        case 0x85: // TUPLE1
          this.stack.push([this.stack.pop()]);
          break;
        case 0x86: { // TUPLE2
          const b = this.stack.pop();
          const a = this.stack.pop();
          this.stack.push([a, b]);
          break;
        }
        case 0x87: { // TUPLE3
          const c = this.stack.pop();
          const b = this.stack.pop();
          const a = this.stack.pop();
          this.stack.push([a, b, c]);
          break;
        }
        case 0x6c: { // LIST
          const listItems = this.stack;
          this.stack = this.metastack.pop() || [];
          this.stack.push(listItems);
          break;
        }
        case 0x64: { // DICT
          const dictItems = this.stack;
          this.stack = this.metastack.pop() || [];
          const dict = new Map<string, unknown>();
          for (let i = 0; i < dictItems.length; i += 2) {
            const key = String(dictItems[i]);
            dict.set(key, dictItems[i + 1]);
          }
          this.stack.push(dict);
          break;
        }
        case 0x73: { // SETITEM
          const value = this.stack.pop();
          const key = String(this.stack.pop());
          const target = this.stack[this.stack.length - 1];
          if (target instanceof Map) {
            target.set(key, value);
          }
          break;
        }
        case 0x75: { // SETITEMS
          const items = this.stack;
          this.stack = this.metastack.pop() || [];
          const target = this.stack[this.stack.length - 1];
          if (target instanceof Map) {
            for (let i = 0; i < items.length; i += 2) {
              const key = String(items[i]);
              target.set(key, items[i + 1]);
            }
          }
          break;
        }
        case 0x65: { // APPENDS
          const items = this.stack;
          this.stack = this.metastack.pop() || [];
          const target = this.stack[this.stack.length - 1];
          if (Array.isArray(target)) {
            target.push(...items);
          }
          break;
        }
        case 0x63: { // GLOBAL
          const module = this.readLine();
          const name = this.readLine();
          this.stack.push({ __global__: `${module}.${name}` });
          break;
        }
        case 0x93: { // STACK_GLOBAL
          const name = this.stack.pop();
          const module = this.stack.pop();
          this.stack.push({ __global__: `${module}.${name}` });
          break;
        }
        case 0x52: { // REDUCE
          const args = this.stack.pop() as unknown[];
          const callable = this.stack.pop() as { __global__?: string };
          // For torch tensors, extract shape info
          if (callable?.__global__?.includes('torch') && Array.isArray(args)) {
            this.stack.push({ __tensor__: true, args });
          } else {
            this.stack.push({ __reduced__: callable?.__global__, args });
          }
          break;
        }
        case 0x81: { // NEWOBJ
          const args = this.stack.pop();
          const cls = this.stack.pop() as { __global__?: string };
          this.stack.push({ __class__: cls?.__global__, __args__: args });
          break;
        }
        case 0x92: { // NEWOBJ_EX
          const kwargs = this.stack.pop();
          const args = this.stack.pop();
          const cls = this.stack.pop() as { __global__?: string };
          this.stack.push({ __class__: cls?.__global__, __args__: args, __kwargs__: kwargs });
          break;
        }
        case 0x62: { // BUILD
          const state = this.stack.pop();
          const obj = this.stack[this.stack.length - 1];
          // Merge state into object
          if (obj && typeof obj === 'object' && state && typeof state === 'object') {
            if (state instanceof Map) {
              for (const [k, v] of state) {
                (obj as Record<string, unknown>)[k] = v;
              }
            } else {
              Object.assign(obj as object, state);
            }
          }
          break;
        }
        case 0x71: { // BINGET
          const idx = this.readByte();
          this.stack.push(this.memo.get(idx));
          break;
        }
        case 0x6a: { // LONG_BINGET
          const idx = this.readUint32();
          this.stack.push(this.memo.get(idx));
          break;
        }
        case 0x68: { // BINPUT
          const idx = this.readByte();
          this.memo.set(idx, this.stack[this.stack.length - 1]);
          break;
        }
        case 0x72: { // LONG_BINPUT
          const idx = this.readUint32();
          this.memo.set(idx, this.stack[this.stack.length - 1]);
          break;
        }
        case 0x94: // MEMOIZE
          this.memo.set(this.memo.size, this.stack[this.stack.length - 1]);
          break;
        case 0x30: // POP
          this.stack.pop();
          break;
        case 0x32: // DUP
          this.stack.push(this.stack[this.stack.length - 1]);
          break;
        case 0x2e: // STOP
          return this.stack[this.stack.length - 1];
        case 0x95: // FRAME
          this.pos += 8;
          break;
        case 0x8e: { // BINBYTES8
          const len = Number(this.data.getBigUint64(this.pos, true));
          this.pos += 8;
          this.stack.push(this.readBytes(len));
          break;
        }
        case 0x43: { // SHORT_BINBYTES
          const len = this.readByte();
          this.stack.push(this.readBytes(len));
          break;
        }
        case 0x44: { // BINBYTES
          const len = this.readUint32();
          this.stack.push(this.readBytes(len));
          break;
        }
        case 0x61: { // APPEND
          const value = this.stack.pop();
          const target = this.stack[this.stack.length - 1];
          if (Array.isArray(target)) {
            target.push(value);
          }
          break;
        }
        case 0x46: // FLOAT
          this.stack.push(parseFloat(this.readLine()));
          break;
        case 0x49: // INT
          this.stack.push(parseInt(this.readLine(), 10));
          break;
        case 0x4c: // LONG
          this.stack.push(parseInt(this.readLine().replace('L', ''), 10));
          break;
        case 0x53: { // STRING
          const line = this.readLine();
          this.stack.push(line.replace(/^['"]|['"]$/g, ''));
          break;
        }
        case 0x56: // UNICODE
          this.stack.push(this.readLine());
          break;
        case 0x70: { // PUT
          const line = this.readLine();
          this.memo.set(parseInt(line, 10), this.stack[this.stack.length - 1]);
          break;
        }
        case 0x67: { // GET
          const line = this.readLine();
          this.stack.push(this.memo.get(parseInt(line, 10)));
          break;
        }
        default:
          // Unknown opcode - skip
          break;
      }
    }
    return this.stack[this.stack.length - 1];
  }
}

/**
 * Recursively collect all string keys from a pickle result
 * This finds all weight/parameter names in the model
 */
function collectAllKeys(obj: unknown, prefix: string = '', depth: number = 0): string[] {
  if (depth > 10) return []; // Prevent infinite recursion
  
  const keys: string[] = [];
  
  if (obj instanceof Map) {
    for (const [key, value] of obj) {
      const fullKey = prefix ? `${prefix}.${key}` : key;
      keys.push(fullKey);
      keys.push(...collectAllKeys(value, fullKey, depth + 1));
    }
  } else if (obj && typeof obj === 'object' && !Array.isArray(obj) && !(obj instanceof Uint8Array)) {
    const o = obj as Record<string, unknown>;
    for (const key of Object.keys(o)) {
      // Skip internal pickle markers
      if (key.startsWith('__')) continue;
      const fullKey = prefix ? `${prefix}.${key}` : key;
      keys.push(fullKey);
      keys.push(...collectAllKeys(o[key], fullKey, depth + 1));
    }
  }
  
  return keys;
}

/**
 * Find the state dict in the pickle result
 * PyTorch models store weights in different structures
 */
function findStateDict(obj: unknown): Map<string, unknown> | Record<string, unknown> | null {
  if (!obj) return null;
  
  // Direct Map (OrderedDict)
  if (obj instanceof Map) {
    // Check if it looks like a state dict
    const keys = Array.from(obj.keys());
    if (keys.some(k => k.includes('.weight') || k.includes('.bias'))) {
      return obj;
    }
    // Check for nested state_dict key
    if (obj.has('state_dict')) {
      return findStateDict(obj.get('state_dict'));
    }
    if (obj.has('model_state_dict')) {
      return findStateDict(obj.get('model_state_dict'));
    }
    if (obj.has('model')) {
      return findStateDict(obj.get('model'));
    }
  }
  
  // Plain object
  if (obj && typeof obj === 'object' && !Array.isArray(obj)) {
    const o = obj as Record<string, unknown>;
    
    // Check specific keys
    if ('state_dict' in o) return findStateDict(o.state_dict);
    if ('model_state_dict' in o) return findStateDict(o.model_state_dict);
    if ('model' in o) return findStateDict(o.model);
    
    // Check if current object looks like a state dict
    const keys = Object.keys(o).filter(k => !k.startsWith('__'));
    if (keys.some(k => k.includes('.weight') || k.includes('.bias'))) {
      return o;
    }
  }
  
  return null;
}

/**
 * Infer layer type from the layer name
 */
function inferLayerType(name: string): LayerType {
  const lower = name.toLowerCase();
  const last = lower.split('.').pop() || '';
  
  // Convolution layers
  if (last.includes('conv') || lower.includes('conv')) {
    if (lower.includes('conv1d') || last === 'conv1d') return 'conv1d';
    if (lower.includes('conv3d') || last === 'conv3d') return 'conv3d';
    if (lower.includes('deconv') || lower.includes('convtranspose')) return 'convTranspose2d';
    return 'conv2d';
  }
  
  // Linear/Dense layers
  if (last === 'fc' || last === 'linear' || last === 'dense' || lower.includes('linear') || lower.includes('fc')) {
    return 'linear';
  }
  
  // Normalization
  if (last.includes('bn') || last.includes('batchnorm') || lower.includes('batchnorm')) {
    if (lower.includes('1d')) return 'batchNorm1d';
    return 'batchNorm2d';
  }
  if (last.includes('ln') || last.includes('layernorm') || lower.includes('layernorm')) return 'layerNorm';
  if (last.includes('groupnorm') || lower.includes('groupnorm')) return 'groupNorm';
  if (last.includes('instancenorm') || lower.includes('instancenorm')) return 'instanceNorm';
  
  // Attention
  if (last.includes('attention') || last.includes('attn') || lower.includes('attention') || lower.includes('attn')) {
    return 'multiHeadAttention';
  }
  if (last === 'q_proj' || last === 'k_proj' || last === 'v_proj' || last === 'o_proj') {
    return 'linear';
  }
  
  // Embedding
  if (last.includes('embed') || lower.includes('embed')) return 'embedding';
  
  // Pooling
  if (last.includes('pool')) {
    if (lower.includes('max')) return 'maxPool2d';
    if (lower.includes('avg') || lower.includes('adaptive')) return 'adaptiveAvgPool';
    return 'maxPool2d';
  }
  
  // Dropout
  if (last.includes('dropout') || lower.includes('dropout')) return 'dropout';
  
  // RNN layers
  if (last === 'lstm' || lower.includes('lstm')) return 'lstm';
  if (last === 'gru' || lower.includes('gru')) return 'gru';
  if (last === 'rnn' || lower.includes('rnn')) return 'rnn';
  
  // Activation hints in name
  if (last === 'relu' || lower.endsWith('_relu')) return 'relu';
  if (last === 'gelu' || lower.endsWith('_gelu')) return 'gelu';
  if (last === 'silu' || last === 'swish') return 'silu';
  
  // MLP/FFN
  if (last === 'mlp' || last === 'ffn' || lower.includes('mlp') || lower.includes('ffn')) {
    return 'linear';
  }
  
  // Default to linear for unknown
  return 'linear';
}

/**
 * Extract layer structure from weight names
 * Uses Netron's approach of grouping by path prefix
 */
function extractLayersFromKeys(keys: string[]): { 
  layers: ExtractedLayer[]; 
  connections: Array<{ source: string; target: string }>;
} {
  const layerMap = new Map<string, ExtractedLayer>();
  
  // Filter to only weight-related keys
  const weightKeys = keys.filter(key => {
    const parts = key.split('.');
    const last = parts[parts.length - 1];
    return ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked',
            'weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0',
            'in_proj_weight', 'in_proj_bias', 'out_proj'].includes(last) ||
           last.endsWith('_weight') || last.endsWith('_bias');
  });
  
  // Parse each weight key to extract layer name
  for (const key of weightKeys) {
    const parts = key.split('.');
    
    // Remove the weight/bias suffix to get layer path
    let layerParts = parts.slice(0, -1);
    
    // Handle nested weight paths like "out_proj.weight"
    if (layerParts.length > 0 && ['out_proj', 'in_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj'].includes(layerParts[layerParts.length - 1])) {
      // Keep these as part of the layer name
    }
    
    const layerName = layerParts.join('.');
    if (!layerName) continue;
    
    const layerId = layerName.replace(/\./g, '_').replace(/[^a-zA-Z0-9_]/g, '');
    
    if (!layerMap.has(layerName)) {
      layerMap.set(layerName, {
        id: layerId,
        name: layerName,
        type: inferLayerType(layerName),
        params: {},
      });
    }
  }
  
  // Sort layers by their natural order (preserving numeric ordering)
  const sortedLayerNames = Array.from(layerMap.keys()).sort((a, b) => {
    // Natural sort for layer names like "layer1.0.conv1", "layer1.1.conv1"
    const aParts = a.split(/(\d+)/).filter(Boolean);
    const bParts = b.split(/(\d+)/).filter(Boolean);
    
    for (let i = 0; i < Math.max(aParts.length, bParts.length); i++) {
      const aVal = aParts[i] ?? '';
      const bVal = bParts[i] ?? '';
      
      const aNum = parseInt(aVal, 10);
      const bNum = parseInt(bVal, 10);
      
      if (!isNaN(aNum) && !isNaN(bNum)) {
        if (aNum !== bNum) return aNum - bNum;
      } else {
        const cmp = aVal.localeCompare(bVal);
        if (cmp !== 0) return cmp;
      }
    }
    return 0;
  });
  
  const layers: ExtractedLayer[] = [];
  const connections: Array<{ source: string; target: string }> = [];
  
  // Add input
  layers.push({ id: 'input', name: 'Input', type: 'input' });
  
  // Add layers in order
  for (const name of sortedLayerNames) {
    layers.push(layerMap.get(name)!);
  }
  
  // Add output
  layers.push({ id: 'output', name: 'Output', type: 'output' });
  
  // Create sequential connections
  for (let i = 0; i < layers.length - 1; i++) {
    connections.push({ source: layers[i].id, target: layers[i + 1].id });
  }
  
  return { layers, connections };
}

/**
 * PyTorch format parser
 */
export const PyTorchParser: FormatParser = {
  extensions: ['.pt', '.pth', '.ckpt', '.bin'],
  
  async canParse(file: File): Promise<boolean> {
    const ext = file.name.toLowerCase();
    return ext.endsWith('.pt') || ext.endsWith('.pth') || ext.endsWith('.ckpt') || ext.endsWith('.bin');
  },
  
  async parse(file: File): Promise<ParseResult> {
    const format = detectFormatFromExtension(file.name);
    const warnings: string[] = [];
    
    try {
      let buffer = await file.arrayBuffer();
      let data = new Uint8Array(buffer);
      
      // Check if it's a ZIP file (PyTorch >= 1.6 format)
      const isZip = data[0] === 0x50 && data[1] === 0x4b;
      
      if (isZip) {
        const result = await parseZipPyTorch(buffer, warnings);
        if (result) {
          return {
            success: true,
            model: result,
            warnings,
            format,
            inferredStructure: true,
          };
        }
      }
      
      // Check if gzip compressed
      if (data[0] === 0x1f && data[1] === 0x8b) {
        try {
          data = pako.ungzip(data);
          buffer = data.buffer as ArrayBuffer;
        } catch {
          warnings.push('Failed to decompress gzip data');
        }
      }
      
      // Parse pickle
      const reader = new PickleReader(buffer);
      const pickleData = reader.parse();
      
      if (!pickleData) {
        throw new Error('Failed to parse pickle data');
      }
      
      // Try to find state dict
      const stateDict = findStateDict(pickleData);
      
      // Collect all keys from the pickle result
      const allKeys = collectAllKeys(stateDict || pickleData);
      
      // Extract layers from keys
      const { layers, connections } = extractLayersFromKeys(allKeys);
      
      if (layers.length <= 2) {
        throw new Error('No layers found in PyTorch model. The file may be corrupted or in an unsupported format.');
      }
      
      const model: NN3DModel = {
        version: '1.0.0',
        metadata: {
          name: file.name.replace(/\.(pt|pth|ckpt|bin)$/i, ''),
          description: `Imported from PyTorch (${layers.length - 2} layers)`,
          framework: 'pytorch',
          created: new Date().toISOString(),
          tags: ['pytorch', 'imported'],
        },
        graph: {
          nodes: layers.map((layer, i) => ({
            id: layer.id,
            type: layer.type as LayerType,
            name: layer.name,
            params: layer.params as Record<string, unknown>,
            depth: i,
          })),
          edges: connections,
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
      
      return {
        success: true,
        model,
        warnings,
        format,
        inferredStructure: true,
      };
      
    } catch (error) {
      console.error('PyTorch parse error:', error);
      return {
        success: false,
        error: `Failed to parse PyTorch file: ${error instanceof Error ? error.message : 'Unknown error'}`,
        warnings,
        format,
        inferredStructure: false,
      };
    }
  }
};

/**
 * Parse PyTorch ZIP format (version >= 1.6)
 */
async function parseZipPyTorch(buffer: ArrayBuffer, warnings: string[]): Promise<NN3DModel | null> {
  try {
    const data = new Uint8Array(buffer);
    const view = new DataView(buffer);
    
    // Find end of central directory
    let eocdOffset = -1;
    for (let i = data.length - 22; i >= 0; i--) {
      if (view.getUint32(i, true) === 0x06054b50) {
        eocdOffset = i;
        break;
      }
    }
    
    if (eocdOffset === -1) {
      warnings.push('Not a valid ZIP file');
      return null;
    }
    
    const cdOffset = view.getUint32(eocdOffset + 16, true);
    const cdEntries = view.getUint16(eocdOffset + 10, true);
    
    // Parse central directory
    let offset = cdOffset;
    const files: Map<string, { offset: number; compressedSize: number; uncompressedSize: number; compression: number }> = new Map();
    
    for (let i = 0; i < cdEntries; i++) {
      if (view.getUint32(offset, true) !== 0x02014b50) break;
      
      const compression = view.getUint16(offset + 10, true);
      const compressedSize = view.getUint32(offset + 20, true);
      const uncompressedSize = view.getUint32(offset + 24, true);
      const nameLen = view.getUint16(offset + 28, true);
      const extraLen = view.getUint16(offset + 30, true);
      const commentLen = view.getUint16(offset + 32, true);
      const localHeaderOffset = view.getUint32(offset + 42, true);
      
      const nameBytes = data.slice(offset + 46, offset + 46 + nameLen);
      const name = new TextDecoder().decode(nameBytes);
      
      files.set(name, { offset: localHeaderOffset, compressedSize, uncompressedSize, compression });
      
      offset += 46 + nameLen + extraLen + commentLen;
    }
    
    // Find data.pkl file
    let pickleFile: { offset: number; compressedSize: number; uncompressedSize: number; compression: number } | undefined;
    
    for (const [name, info] of files) {
      if (name.endsWith('data.pkl') || name.endsWith('/data.pkl')) {
        pickleFile = info;
        break;
      }
    }
    
    if (!pickleFile) {
      for (const [name, info] of files) {
        if (name.endsWith('.pkl')) {
          pickleFile = info;
          break;
        }
      }
    }
    
    if (!pickleFile) {
      warnings.push('No pickle file found in PyTorch archive');
      return null;
    }
    
    // Read local file header
    const localOffset = pickleFile.offset;
    if (view.getUint32(localOffset, true) !== 0x04034b50) {
      warnings.push('Invalid local file header');
      return null;
    }
    
    const localNameLen = view.getUint16(localOffset + 26, true);
    const localExtraLen = view.getUint16(localOffset + 28, true);
    const dataOffset = localOffset + 30 + localNameLen + localExtraLen;
    
    let fileData = data.slice(dataOffset, dataOffset + pickleFile.compressedSize);
    
    // Decompress if needed
    if (pickleFile.compression === 8) {
      try {
        fileData = pako.inflateRaw(fileData);
      } catch {
        try {
          fileData = pako.inflate(fileData);
        } catch {
          warnings.push('Failed to decompress ZIP entry');
          return null;
        }
      }
    }
    
    // Parse pickle
    const reader = new PickleReader(fileData.buffer as ArrayBuffer);
    const pickleData = reader.parse();
    
    if (!pickleData) {
      return null;
    }
    
    const stateDict = findStateDict(pickleData);
    const allKeys = collectAllKeys(stateDict || pickleData);
    const { layers, connections } = extractLayersFromKeys(allKeys);
    
    if (layers.length <= 2) {
      return null;
    }
    
    return {
      version: '1.0.0',
      metadata: {
        name: 'PyTorch Model',
        description: `Imported from PyTorch ZIP (${layers.length - 2} layers)`,
        framework: 'pytorch',
        created: new Date().toISOString(),
        tags: ['pytorch', 'imported'],
      },
      graph: {
        nodes: layers.map((layer, i) => ({
          id: layer.id,
          type: layer.type as LayerType,
          name: layer.name,
          params: layer.params as Record<string, unknown>,
          depth: i,
        })),
        edges: connections,
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
    
  } catch (error) {
    warnings.push(`ZIP parse error: ${error instanceof Error ? error.message : 'Unknown'}`);
    return null;
  }
}

/**
 * Create a placeholder model for unsupported formats
 */
export function createPlaceholderModel(filename: string, format: string): NN3DModel {
  return {
    version: '1.0.0',
    metadata: {
      name: filename,
      description: `Unable to parse ${format} format directly.`,
      framework: 'pytorch',
      created: new Date().toISOString(),
    },
    graph: {
      nodes: [
        {
          id: 'unsupported',
          type: 'custom',
          name: `${format} model`,
          depth: 0,
        },
      ],
      edges: [],
    },
    visualization: {
      layout: 'layered',
      theme: 'dark',
    },
  };
}

export default PyTorchParser;
