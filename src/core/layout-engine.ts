/**
 * Deterministic Layout Engine
 * 
 * Positions nodes in 3D space following a consistent, predictable algorithm.
 * Same model always produces the same layout.
 * 
 * Coordinate System:
 * - X → Forward (data flow direction)
 * - Y → Vertical (depth/resolution/blocks)
 * - Z → Width (channels/grouping)
 */

import type { HierarchyNode, ModelHierarchy, HierarchyConnection } from './model-hierarchy';
import { calculateLayerDimensions, type LayerDimensions } from './layer-geometry';

// ============================================================================
// Layout Configuration
// ============================================================================

export interface LayoutConfig {
  // Spacing
  macroSpacing: number;      // Space between encoder/decoder/head
  stageSpacing: number;      // Space between stages within macro
  layerSpacing: number;      // Space between individual layers
  blockSpacing: number;      // Space between blocks within stage
  
  // Offsets
  baseY: number;             // Starting Y position
  channelZScale: number;     // How much Z offset based on channel size
  
  // Grouping
  groupPadding: number;      // Padding inside group containers
  
  // Animation
  transitionDuration: number;
}

const DEFAULT_LAYOUT_CONFIG: LayoutConfig = {
  macroSpacing: 8.0,
  stageSpacing: 4.0,
  layerSpacing: 1.2,
  blockSpacing: 0.8,
  baseY: 0,
  channelZScale: 0.002,
  groupPadding: 0.5,
  transitionDuration: 500,
};

// ============================================================================
// Position Types
// ============================================================================

export interface Position3D {
  x: number;
  y: number;
  z: number;
}

export interface NodeLayout {
  id: string;
  position: Position3D;
  dimensions: LayerDimensions;
  rotation: { x: number; y: number; z: number };
  visible: boolean;
  expanded: boolean;
}

export interface ConnectionLayout {
  id: string;
  sourcePosition: Position3D;
  targetPosition: Position3D;
  controlPoints: Position3D[];  // For curved connections
  isSkipConnection: boolean;
}

export interface LayoutResult {
  nodes: Map<string, NodeLayout>;
  connections: ConnectionLayout[];
  bounds: {
    min: Position3D;
    max: Position3D;
    center: Position3D;
  };
  cameraSuggestion: {
    position: Position3D;
    target: Position3D;
  };
}

// ============================================================================
// Layout State (for expand/collapse)
// ============================================================================

export interface LayoutState {
  expandedNodes: Set<string>;
  currentLevel: 1 | 2 | 3;
  focusedNodeId: string | null;
}

const defaultLayoutState: LayoutState = {
  expandedNodes: new Set(),
  currentLevel: 1,
  focusedNodeId: null,
};

// ============================================================================
// Main Layout Engine
// ============================================================================

export class LayoutEngine {
  private config: LayoutConfig;
  private state: LayoutState;
  private hierarchy: ModelHierarchy | null = null;
  private cachedLayout: LayoutResult | null = null;
  
  constructor(config: Partial<LayoutConfig> = {}) {
    this.config = { ...DEFAULT_LAYOUT_CONFIG, ...config };
    this.state = { ...defaultLayoutState };
  }
  
  /**
   * Set the model hierarchy to layout
   */
  setHierarchy(hierarchy: ModelHierarchy): void {
    this.hierarchy = hierarchy;
    this.cachedLayout = null;
  }
  
  /**
   * Update layout state
   */
  updateState(updates: Partial<LayoutState>): void {
    this.state = { ...this.state, ...updates };
    this.cachedLayout = null;
  }
  
  /**
   * Toggle node expansion
   */
  toggleExpanded(nodeId: string): void {
    const expanded = new Set(this.state.expandedNodes);
    if (expanded.has(nodeId)) {
      expanded.delete(nodeId);
    } else {
      expanded.add(nodeId);
    }
    this.updateState({ expandedNodes: expanded });
  }
  
  /**
   * Set current view level
   */
  setLevel(level: 1 | 2 | 3): void {
    this.updateState({ currentLevel: level });
  }
  
  /**
   * Compute layout for current state
   */
  computeLayout(): LayoutResult {
    if (this.cachedLayout) return this.cachedLayout;
    if (!this.hierarchy) {
      return this.createEmptyLayout();
    }
    
    const nodes = new Map<string, NodeLayout>();
    const connections: ConnectionLayout[] = [];
    
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    
    // Layout based on current level
    const { currentLevel, expandedNodes } = this.state;
    
    // Position macro nodes (Level 1)
    let macroX = 0;
    
    this.hierarchy.macroNodes.forEach((macroNode) => {
      const macroDims = calculateLayerDimensions(macroNode);
      const isExpanded = expandedNodes.has(macroNode.id) || currentLevel > 1;
      
      if (currentLevel === 1 && !isExpanded) {
        // Show macro as single block
        const position: Position3D = {
          x: macroX,
          y: this.config.baseY,
          z: 0,
        };
        
        nodes.set(macroNode.id, {
          id: macroNode.id,
          position,
          dimensions: macroDims,
          rotation: { x: 0, y: 0, z: 0 },
          visible: true,
          expanded: false,
        });
        
        this.updateBounds(position, macroDims);
        macroX += macroDims.depth + this.config.macroSpacing;
      } else {
        // Expand macro to show stages/layers
        const childLayout = this.layoutChildren(
          macroNode,
          { x: macroX, y: this.config.baseY, z: 0 },
          currentLevel,
          expandedNodes
        );
        
        childLayout.forEach((layout, id) => {
          nodes.set(id, layout);
          this.updateBoundsFromLayout(layout);
        });
        
        // Calculate macro extent
        let macroMaxX = macroX;
        childLayout.forEach(layout => {
          macroMaxX = Math.max(macroMaxX, layout.position.x + layout.dimensions.depth);
        });
        
        macroX = macroMaxX + this.config.macroSpacing;
      }
    });
    
    // Layout connections
    this.hierarchy.connections.forEach(conn => {
      const sourceLayout = nodes.get(conn.source);
      const targetLayout = nodes.get(conn.target);
      
      if (sourceLayout && targetLayout && sourceLayout.visible && targetLayout.visible) {
        connections.push(this.layoutConnection(conn, sourceLayout, targetLayout));
      }
    });
    
    // Calculate bounds
    nodes.forEach(layout => {
      minX = Math.min(minX, layout.position.x - layout.dimensions.depth / 2);
      maxX = Math.max(maxX, layout.position.x + layout.dimensions.depth / 2);
      minY = Math.min(minY, layout.position.y - layout.dimensions.height / 2);
      maxY = Math.max(maxY, layout.position.y + layout.dimensions.height / 2);
      minZ = Math.min(minZ, layout.position.z - layout.dimensions.width / 2);
      maxZ = Math.max(maxZ, layout.position.z + layout.dimensions.width / 2);
    });
    
    const bounds = {
      min: { x: minX, y: minY, z: minZ },
      max: { x: maxX, y: maxY, z: maxZ },
      center: {
        x: (minX + maxX) / 2,
        y: (minY + maxY) / 2,
        z: (minZ + maxZ) / 2,
      },
    };
    
    // Calculate camera suggestion
    const extent = Math.max(maxX - minX, maxY - minY, maxZ - minZ);
    const cameraSuggestion = {
      position: {
        x: bounds.center.x - extent * 0.5,
        y: bounds.center.y + extent * 0.5,
        z: bounds.center.z + extent * 1.2,
      },
      target: bounds.center,
    };
    
    this.cachedLayout = { nodes, connections, bounds, cameraSuggestion };
    return this.cachedLayout;
  }
  
  /**
   * Layout children of a node
   */
  private layoutChildren(
    parentNode: HierarchyNode,
    startPos: Position3D,
    targetLevel: 1 | 2 | 3,
    expandedNodes: Set<string>
  ): Map<string, NodeLayout> {
    const layouts = new Map<string, NodeLayout>();
    
    let currentX = startPos.x;
    let currentY = startPos.y;
    let blockIndex = 0;
    
    parentNode.children.forEach((child) => {
      const childDims = calculateLayerDimensions(child);
      const isExpanded = expandedNodes.has(child.id) || (child.level < targetLevel);
      
      // Calculate Z position based on channel size (creates depth effect)
      const channelOffset = (child.channelSize || 64) * this.config.channelZScale;
      
      if (child.level >= targetLevel || child.children.length === 0 || !isExpanded) {
        // Render this node
        const position: Position3D = {
          x: currentX,
          y: currentY + (child.blockIndex || 0) * this.config.blockSpacing,
          z: startPos.z + channelOffset,
        };
        
        layouts.set(child.id, {
          id: child.id,
          position,
          dimensions: childDims,
          rotation: { x: 0, y: 0, z: 0 },
          visible: true,
          expanded: isExpanded,
        });
        
        currentX += childDims.depth + this.config.layerSpacing;
      } else {
        // Recursively layout children
        const childLayouts = this.layoutChildren(
          child,
          { x: currentX, y: currentY, z: startPos.z },
          targetLevel,
          expandedNodes
        );
        
        childLayouts.forEach((layout, id) => {
          layouts.set(id, layout);
          currentX = Math.max(currentX, layout.position.x + layout.dimensions.depth);
        });
        
        currentX += this.config.stageSpacing;
      }
      
      // Track block changes for Y positioning
      if (child.blockIndex !== undefined && child.blockIndex !== blockIndex) {
        blockIndex = child.blockIndex;
      }
    });
    
    return layouts;
  }
  
  /**
   * Layout a connection between two nodes
   */
  private layoutConnection(
    conn: HierarchyConnection,
    source: NodeLayout,
    target: NodeLayout
  ): ConnectionLayout {
    // Source point: right side of source node
    const sourcePos: Position3D = {
      x: source.position.x + source.dimensions.depth / 2,
      y: source.position.y,
      z: source.position.z,
    };
    
    // Target point: left side of target node
    const targetPos: Position3D = {
      x: target.position.x - target.dimensions.depth / 2,
      y: target.position.y,
      z: target.position.z,
    };
    
    // Calculate control points for curves
    const controlPoints: Position3D[] = [];
    
    if (conn.isSkipConnection) {
      // Arc above the main flow
      const midX = (sourcePos.x + targetPos.x) / 2;
      const arcHeight = Math.abs(target.position.x - source.position.x) * 0.3;
      
      controlPoints.push({
        x: midX,
        y: Math.max(sourcePos.y, targetPos.y) + arcHeight,
        z: (sourcePos.z + targetPos.z) / 2,
      });
    } else if (Math.abs(sourcePos.z - targetPos.z) > 0.5) {
      // Z-axis curve for channel transitions
      const midX = (sourcePos.x + targetPos.x) / 2;
      controlPoints.push({
        x: midX,
        y: (sourcePos.y + targetPos.y) / 2,
        z: (sourcePos.z + targetPos.z) / 2,
      });
    }
    
    return {
      id: conn.id,
      sourcePosition: sourcePos,
      targetPosition: targetPos,
      controlPoints,
      isSkipConnection: conn.isSkipConnection,
    };
  }
  
  private updateBounds(_pos: Position3D, _dims: LayerDimensions): void {
    // Helper for bound tracking - reserved for future use
  }
  
  private updateBoundsFromLayout(_layout: NodeLayout): void {
    // Helper for bound tracking - reserved for future use
  }
  
  private createEmptyLayout(): LayoutResult {
    return {
      nodes: new Map(),
      connections: [],
      bounds: {
        min: { x: 0, y: 0, z: 0 },
        max: { x: 0, y: 0, z: 0 },
        center: { x: 0, y: 0, z: 0 },
      },
      cameraSuggestion: {
        position: { x: -10, y: 10, z: 20 },
        target: { x: 0, y: 0, z: 0 },
      },
    };
  }
}

// ============================================================================
// Standalone Layout Functions
// ============================================================================

/**
 * Compute full layout for a model hierarchy
 */
export function computeFullLayout(
  hierarchy: ModelHierarchy,
  level: 1 | 2 | 3 = 3,
  config: Partial<LayoutConfig> = {}
): LayoutResult {
  const engine = new LayoutEngine(config);
  engine.setHierarchy(hierarchy);
  engine.setLevel(level);
  return engine.computeLayout();
}

/**
 * Compute layout for a specific node and its descendants
 */
export function computeSubLayout(
  hierarchy: ModelHierarchy,
  nodeId: string,
  config: Partial<LayoutConfig> = {}
): LayoutResult {
  const engine = new LayoutEngine(config);
  engine.setHierarchy(hierarchy);
  engine.updateState({ 
    expandedNodes: new Set([nodeId]),
    currentLevel: 3,
    focusedNodeId: nodeId,
  });
  return engine.computeLayout();
}

/**
 * Create a layout that focuses on data flow through the network
 */
export function computeFlowLayout(
  hierarchy: ModelHierarchy,
  config: Partial<LayoutConfig> = {}
): LayoutResult {
  const mergedConfig: LayoutConfig = {
    ...DEFAULT_LAYOUT_CONFIG,
    ...config,
    // Tighter spacing for flow visualization
    layerSpacing: 0.8,
    stageSpacing: 2.0,
  };
  
  const engine = new LayoutEngine(mergedConfig);
  engine.setHierarchy(hierarchy);
  engine.setLevel(3);
  return engine.computeLayout();
}
