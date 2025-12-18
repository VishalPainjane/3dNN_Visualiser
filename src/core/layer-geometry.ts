/**
 * Layer Geometry System
 * 
 * Generates 3D geometries for different layer types with semantic sizing.
 * 
 * Size Encoding:
 * - Width (Z) → Channels
 * - Height (Y) → Spatial resolution (relative)
 * - Depth (X) → Constant for flow clarity
 */

import * as THREE from 'three';
import type { HierarchyNode } from './model-hierarchy';

// ============================================================================
// Color System (Category-based, semantic)
// ============================================================================

export const LAYER_COLORS: Record<string, string> = {
  // Core layers
  convolution: '#4A90D9',      // Blue
  linear: '#9B59B6',           // Purple
  normalization: '#2ECC71',    // Green
  activation: '#F39C12',       // Orange
  pooling: '#1ABC9C',          // Teal
  
  // Special layers
  recurrent: '#E74C3C',        // Red
  attention: '#E91E63',        // Pink
  embedding: '#00BCD4',        // Cyan
  regularization: '#95A5A6',   // Gray
  reshape: '#607D8B',          // Blue Gray
  
  // Macro categories
  encoder: '#3498DB',          // Bright Blue
  decoder: '#9B59B6',          // Purple
  output: '#E74C3C',           // Red
  input: '#2ECC71',            // Green
  
  // Grouping
  stage: '#34495E',            // Dark Gray
  group: '#2C3E50',            // Darker Gray
  
  // Default
  other: '#7F8C8D',            // Gray
};

export const LAYER_COLORS_DARK: Record<string, string> = {
  convolution: '#2E5A8A',
  linear: '#6E3D8A',
  normalization: '#1D8A4A',
  activation: '#B87410',
  pooling: '#128A74',
  recurrent: '#A83232',
  attention: '#A31545',
  embedding: '#008A9A',
  regularization: '#6B7B7B',
  reshape: '#455A64',
  encoder: '#236B99',
  decoder: '#6E3D8A',
  output: '#A83232',
  input: '#1D8A4A',
  stage: '#2A3F50',
  group: '#1E2D3A',
  other: '#5A6A6D',
};

export function getLayerColor(category: string, variant: 'light' | 'dark' = 'light'): string {
  const colors = variant === 'light' ? LAYER_COLORS : LAYER_COLORS_DARK;
  return colors[category] || colors.other;
}

// ============================================================================
// Size Calculation
// ============================================================================

const SIZE_CONFIG = {
  // Base sizes
  minWidth: 0.3,
  maxWidth: 2.0,
  minHeight: 0.2,
  maxHeight: 1.5,
  baseDepth: 0.4,
  
  // Scaling factors
  channelScale: 0.003, // Logarithmic scale for channels
  paramScale: 0.00001, // Scale for parameter count
  
  // Group sizes
  macroHeight: 2.0,
  stageHeight: 1.5,
  
  // Normalization
  maxChannels: 2048,
  maxParams: 10000000,
};

export interface LayerDimensions {
  width: number;   // Z-axis (channels)
  height: number;  // Y-axis (spatial/block)
  depth: number;   // X-axis (flow direction)
}

/**
 * Calculate layer dimensions based on channels and parameters
 */
export function calculateLayerDimensions(node: HierarchyNode): LayerDimensions {
  const channels = node.channelSize || 64;
  const params = node.totalParams || 0;
  
  // Logarithmic scaling for width (channels)
  const normalizedChannels = Math.log2(Math.max(channels, 1)) / Math.log2(SIZE_CONFIG.maxChannels);
  const width = SIZE_CONFIG.minWidth + normalizedChannels * (SIZE_CONFIG.maxWidth - SIZE_CONFIG.minWidth);
  
  // Height based on layer type and parameters
  let height: number;
  if (node.level === 1) {
    height = SIZE_CONFIG.macroHeight;
  } else if (node.level === 2) {
    height = SIZE_CONFIG.stageHeight;
  } else {
    // For individual layers, use parameter count
    const normalizedParams = Math.log10(Math.max(params, 1)) / Math.log10(SIZE_CONFIG.maxParams);
    height = SIZE_CONFIG.minHeight + normalizedParams * (SIZE_CONFIG.maxHeight - SIZE_CONFIG.minHeight);
  }
  
  return {
    width: Math.max(SIZE_CONFIG.minWidth, width),
    height: Math.max(SIZE_CONFIG.minHeight, height),
    depth: SIZE_CONFIG.baseDepth,
  };
}

// ============================================================================
// Geometry Generators
// ============================================================================

export type LayerGeometryType = 
  | 'box'           // Default, linear layers
  | 'prism'         // Convolution layers
  | 'plate'         // Normalization layers
  | 'cylinder'      // Activation layers
  | 'hexagon'       // Attention layers
  | 'pyramid'       // Output/head layers
  | 'rounded-box'   // Pooling layers
  | 'container';    // Group containers

/**
 * Get geometry type for a layer category
 */
export function getGeometryType(category: string): LayerGeometryType {
  const mapping: Record<string, LayerGeometryType> = {
    convolution: 'prism',
    linear: 'box',
    normalization: 'plate',
    activation: 'cylinder',
    pooling: 'rounded-box',
    recurrent: 'hexagon',
    attention: 'hexagon',
    embedding: 'box',
    regularization: 'plate',
    reshape: 'plate',
    output: 'pyramid',
    encoder: 'container',
    decoder: 'container',
    stage: 'container',
    group: 'container',
  };
  
  return mapping[category] || 'box';
}

/**
 * Create box geometry (default)
 */
export function createBoxGeometry(dims: LayerDimensions): THREE.BufferGeometry {
  return new THREE.BoxGeometry(dims.depth, dims.height, dims.width);
}

/**
 * Create prism geometry (for convolution layers)
 */
export function createPrismGeometry(dims: LayerDimensions): THREE.BufferGeometry {
  // Slightly tapered box to indicate transformation
  const shape = new THREE.Shape();
  const hw = dims.width / 2;
  const hd = dims.depth / 2;
  const taper = 0.1;
  
  shape.moveTo(-hd, -hw * (1 + taper));
  shape.lineTo(hd, -hw);
  shape.lineTo(hd, hw);
  shape.lineTo(-hd, hw * (1 + taper));
  shape.closePath();
  
  const extrudeSettings = {
    depth: dims.height,
    bevelEnabled: false,
  };
  
  const geometry = new THREE.ExtrudeGeometry(shape, extrudeSettings);
  geometry.rotateX(-Math.PI / 2);
  geometry.translate(0, dims.height / 2, 0);
  
  return geometry;
}

/**
 * Create thin plate geometry (for normalization/reshape)
 */
export function createPlateGeometry(dims: LayerDimensions): THREE.BufferGeometry {
  return new THREE.BoxGeometry(dims.depth * 0.3, dims.height, dims.width);
}

/**
 * Create cylinder geometry (for activation layers)
 */
export function createCylinderGeometry(dims: LayerDimensions): THREE.BufferGeometry {
  return new THREE.CylinderGeometry(
    dims.width / 2.5,
    dims.width / 2.5,
    dims.height,
    16
  );
}

/**
 * Create hexagonal prism (for attention/recurrent layers)
 */
export function createHexagonGeometry(dims: LayerDimensions): THREE.BufferGeometry {
  const shape = new THREE.Shape();
  const radius = dims.width / 2;
  
  for (let i = 0; i < 6; i++) {
    const angle = (Math.PI / 3) * i - Math.PI / 2;
    const x = Math.cos(angle) * radius;
    const y = Math.sin(angle) * radius;
    
    if (i === 0) {
      shape.moveTo(x, y);
    } else {
      shape.lineTo(x, y);
    }
  }
  shape.closePath();
  
  const extrudeSettings = {
    depth: dims.depth,
    bevelEnabled: false,
  };
  
  const geometry = new THREE.ExtrudeGeometry(shape, extrudeSettings);
  geometry.rotateY(Math.PI / 2);
  geometry.rotateZ(Math.PI / 2);
  
  return geometry;
}

/**
 * Create pyramid geometry (for output layers)
 */
export function createPyramidGeometry(dims: LayerDimensions): THREE.BufferGeometry {
  return new THREE.ConeGeometry(dims.width / 2, dims.height, 4);
}

/**
 * Create rounded box geometry (for pooling layers)
 */
export function createRoundedBoxGeometry(dims: LayerDimensions): THREE.BufferGeometry {
  // Simple approach: regular box with slight modifications
  // For true rounded corners, use RoundedBoxGeometry from drei or custom
  const geometry = new THREE.BoxGeometry(
    dims.depth,
    dims.height,
    dims.width,
    2, 2, 2
  );
  
  return geometry;
}

/**
 * Create container geometry (for groups - wireframe style)
 */
export function createContainerGeometry(dims: LayerDimensions): THREE.BufferGeometry {
  return new THREE.BoxGeometry(
    dims.depth * 2,
    dims.height,
    dims.width * 1.5
  );
}

/**
 * Main geometry factory
 */
export function createLayerGeometry(
  node: HierarchyNode,
  dims?: LayerDimensions
): THREE.BufferGeometry {
  const dimensions = dims || calculateLayerDimensions(node);
  const geometryType = getGeometryType(node.category);
  
  switch (geometryType) {
    case 'prism':
      return createPrismGeometry(dimensions);
    case 'plate':
      return createPlateGeometry(dimensions);
    case 'cylinder':
      return createCylinderGeometry(dimensions);
    case 'hexagon':
      return createHexagonGeometry(dimensions);
    case 'pyramid':
      return createPyramidGeometry(dimensions);
    case 'rounded-box':
      return createRoundedBoxGeometry(dimensions);
    case 'container':
      return createContainerGeometry(dimensions);
    case 'box':
    default:
      return createBoxGeometry(dimensions);
  }
}

// ============================================================================
// Material Generators
// ============================================================================

export interface LayerMaterialOptions {
  color: string;
  opacity?: number;
  wireframe?: boolean;
  selected?: boolean;
  hovered?: boolean;
  isContainer?: boolean;
}

/**
 * Create material for layer visualization
 */
export function createLayerMaterial(options: LayerMaterialOptions): THREE.Material {
  const { color, opacity = 1, wireframe = false, selected = false, hovered = false, isContainer = false } = options;
  
  if (isContainer) {
    return new THREE.MeshBasicMaterial({
      color: new THREE.Color(color),
      transparent: true,
      opacity: 0.15,
      wireframe: true,
    });
  }
  
  const baseColor = new THREE.Color(color);
  
  if (selected) {
    baseColor.multiplyScalar(1.3);
  } else if (hovered) {
    baseColor.multiplyScalar(1.15);
  }
  
  return new THREE.MeshStandardMaterial({
    color: baseColor,
    transparent: opacity < 1,
    opacity,
    wireframe,
    metalness: 0.1,
    roughness: 0.7,
    emissive: selected ? baseColor.clone().multiplyScalar(0.2) : undefined,
  });
}

/**
 * Create edge/outline material
 */
export function createEdgeMaterial(color: string): THREE.LineBasicMaterial {
  return new THREE.LineBasicMaterial({
    color: new THREE.Color(color),
    linewidth: 2,
  });
}

// ============================================================================
// Geometry Cache (for InstancedMesh optimization)
// ============================================================================

const geometryCache = new Map<string, THREE.BufferGeometry>();

export function getCachedGeometry(
  node: HierarchyNode,
  dims: LayerDimensions
): THREE.BufferGeometry {
  const key = `${node.category}_${dims.width.toFixed(2)}_${dims.height.toFixed(2)}_${dims.depth.toFixed(2)}`;
  
  if (!geometryCache.has(key)) {
    geometryCache.set(key, createLayerGeometry(node, dims));
  }
  
  return geometryCache.get(key)!;
}

export function clearGeometryCache(): void {
  geometryCache.forEach(geometry => geometry.dispose());
  geometryCache.clear();
}
