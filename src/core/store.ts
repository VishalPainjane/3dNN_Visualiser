import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type { NN3DModel, NN3DNode, NN3DEdge, VisualizationConfig, Position3D } from '@/schema/types';

/**
 * Computed node data for rendering
 */
export interface ComputedNode extends NN3DNode {
  computedPosition: Position3D;
  color: string;
  scale: { x: number; y: number; z: number };
  visible: boolean;
  selected: boolean;
  hovered: boolean;
  lod: number; // 0 = high detail, 1 = medium, 2 = low
}

/**
 * Computed edge data for rendering
 */
export interface ComputedEdge extends NN3DEdge {
  sourcePosition: Position3D;
  targetPosition: Position3D;
  color: string;
  visible: boolean;
  highlighted: boolean;
}

/**
 * Camera state
 */
export interface CameraState {
  position: Position3D;
  target: Position3D;
  zoom: number;
}

/**
 * Selection state
 */
export interface SelectionState {
  selectedNodeId: string | null;
  hoveredNodeId: string | null;
  selectedEdgeId: string | null;
}

/**
 * Visualization store state
 */
export interface VisualizerState {
  // Model data
  model: NN3DModel | null;
  isLoading: boolean;
  error: string | null;
  
  // Computed data for rendering
  computedNodes: Map<string, ComputedNode>;
  computedEdges: ComputedEdge[];
  
  // View state
  camera: CameraState;
  selection: SelectionState;
  
  // Configuration
  config: VisualizationConfig;
  
  // Actions
  loadModel: (model: NN3DModel) => void;
  clearModel: () => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  
  // Selection actions
  selectNode: (nodeId: string | null) => void;
  hoverNode: (nodeId: string | null) => void;
  selectEdge: (edgeId: string | null) => void;
  
  // Camera actions
  setCameraPosition: (position: Position3D) => void;
  setCameraTarget: (target: Position3D) => void;
  resetCamera: () => void;
  
  // Config actions
  updateConfig: (config: Partial<VisualizationConfig>) => void;
  
  // Computed data actions
  updateNodePositions: (positions: Map<string, Position3D>) => void;
  updateNodeLOD: (lodMap: Map<string, number>) => void;
}

// Default configuration
const DEFAULT_CONFIG: VisualizationConfig = {
  layout: 'layered',
  theme: 'dark',
  layerSpacing: 3.0,
  nodeScale: 1.0,
  showLabels: true,
  showEdges: true,
  edgeStyle: 'tube',
};

// Default camera state
const DEFAULT_CAMERA: CameraState = {
  position: { x: 0, y: 5, z: 20 },
  target: { x: 0, y: 0, z: 0 },
  zoom: 1,
};

/**
 * Main visualizer store
 */
export const useVisualizerStore = create<VisualizerState>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
    model: null,
    isLoading: false,
    error: null,
    computedNodes: new Map(),
    computedEdges: [],
    camera: DEFAULT_CAMERA,
    selection: {
      selectedNodeId: null,
      hoveredNodeId: null,
      selectedEdgeId: null,
    },
    config: DEFAULT_CONFIG,
    
    // Model actions
    loadModel: (model: NN3DModel) => {
      const config = { ...DEFAULT_CONFIG, ...model.visualization };
      
      // Initialize computed nodes
      const computedNodes = new Map<string, ComputedNode>();
      model.graph.nodes.forEach((node, index) => {
        computedNodes.set(node.id, {
          ...node,
          computedPosition: node.position || { x: 0, y: index * config.layerSpacing!, z: 0 },
          color: getNodeColor(node.type, config),
          scale: calculateNodeScale(node, config),
          visible: true,
          selected: false,
          hovered: false,
          lod: 0,
        });
      });
      
      // Initialize computed edges
      const computedEdges = model.graph.edges.map((edge, index) => {
        const sourceNode = computedNodes.get(edge.source);
        const targetNode = computedNodes.get(edge.target);
        
        return {
          ...edge,
          id: edge.id || `edge-${index}`,
          sourcePosition: sourceNode?.computedPosition || { x: 0, y: 0, z: 0 },
          targetPosition: targetNode?.computedPosition || { x: 0, y: 0, z: 0 },
          color: '#888888',
          visible: true,
          highlighted: false,
        };
      });
      
      set({
        model,
        config,
        computedNodes,
        computedEdges,
        isLoading: false,
        error: null,
      });
    },
    
    clearModel: () => {
      set({
        model: null,
        computedNodes: new Map(),
        computedEdges: [],
        selection: {
          selectedNodeId: null,
          hoveredNodeId: null,
          selectedEdgeId: null,
        },
      });
    },
    
    setLoading: (isLoading: boolean) => set({ isLoading }),
    setError: (error: string | null) => set({ error, isLoading: false }),
    
    // Selection actions
    selectNode: (nodeId: string | null) => {
      const { computedNodes, computedEdges, selection } = get();
      
      // Update previous selection
      if (selection.selectedNodeId) {
        const prevNode = computedNodes.get(selection.selectedNodeId);
        if (prevNode) {
          computedNodes.set(selection.selectedNodeId, { ...prevNode, selected: false });
        }
      }
      
      // Update new selection
      if (nodeId) {
        const node = computedNodes.get(nodeId);
        if (node) {
          computedNodes.set(nodeId, { ...node, selected: true });
        }
      }
      
      // Highlight connected edges
      const updatedEdges = computedEdges.map(edge => ({
        ...edge,
        highlighted: nodeId ? (edge.source === nodeId || edge.target === nodeId) : false,
      }));
      
      set({
        computedNodes: new Map(computedNodes),
        computedEdges: updatedEdges,
        selection: { ...selection, selectedNodeId: nodeId },
      });
    },
    
    hoverNode: (nodeId: string | null) => {
      const { computedNodes, selection } = get();
      
      // Update previous hover
      if (selection.hoveredNodeId && selection.hoveredNodeId !== nodeId) {
        const prevNode = computedNodes.get(selection.hoveredNodeId);
        if (prevNode) {
          computedNodes.set(selection.hoveredNodeId, { ...prevNode, hovered: false });
        }
      }
      
      // Update new hover
      if (nodeId) {
        const node = computedNodes.get(nodeId);
        if (node) {
          computedNodes.set(nodeId, { ...node, hovered: true });
        }
      }
      
      set({
        computedNodes: new Map(computedNodes),
        selection: { ...selection, hoveredNodeId: nodeId },
      });
    },
    
    selectEdge: (edgeId: string | null) => {
      set(state => ({
        selection: { ...state.selection, selectedEdgeId: edgeId },
      }));
    },
    
    // Camera actions
    setCameraPosition: (position: Position3D) => {
      set(state => ({
        camera: { ...state.camera, position },
      }));
    },
    
    setCameraTarget: (target: Position3D) => {
      set(state => ({
        camera: { ...state.camera, target },
      }));
    },
    
    resetCamera: () => set({ camera: DEFAULT_CAMERA }),
    
    // Config actions
    updateConfig: (configUpdate: Partial<VisualizationConfig>) => {
      set(state => ({
        config: { ...state.config, ...configUpdate },
      }));
    },
    
    // Computed data actions
    updateNodePositions: (positions: Map<string, Position3D>) => {
      const { computedNodes, computedEdges } = get();
      
      positions.forEach((position, nodeId) => {
        const node = computedNodes.get(nodeId);
        if (node) {
          computedNodes.set(nodeId, { ...node, computedPosition: position });
        }
      });
      
      // Update edge positions
      const updatedEdges = computedEdges.map(edge => {
        const sourceNode = computedNodes.get(edge.source);
        const targetNode = computedNodes.get(edge.target);
        return {
          ...edge,
          sourcePosition: sourceNode?.computedPosition || edge.sourcePosition,
          targetPosition: targetNode?.computedPosition || edge.targetPosition,
        };
      });
      
      set({
        computedNodes: new Map(computedNodes),
        computedEdges: updatedEdges,
      });
    },
    
    updateNodeLOD: (lodMap: Map<string, number>) => {
      const { computedNodes } = get();
      
      lodMap.forEach((lod, nodeId) => {
        const node = computedNodes.get(nodeId);
        if (node) {
          computedNodes.set(nodeId, { ...node, lod });
        }
      });
      
      set({ computedNodes: new Map(computedNodes) });
    },
  }))
);

// Helper functions
import { LAYER_CATEGORIES, DEFAULT_CATEGORY_COLORS, LayerType } from '@/schema/types';

function getNodeColor(layerType: LayerType, config: VisualizationConfig): string {
  const category = LAYER_CATEGORIES[layerType] || 'other';
  return config.colorScheme?.[layerType] || 
         config.colorScheme?.[category] || 
         DEFAULT_CATEGORY_COLORS[category];
}

function calculateNodeScale(node: NN3DNode, config: VisualizationConfig): { x: number; y: number; z: number } {
  const baseScale = config.nodeScale || 1.0;
  
  // Scale based on output shape if available
  if (node.outputShape && node.outputShape.length > 0) {
    const dims = node.outputShape.filter((d): d is number => typeof d === 'number');
    if (dims.length >= 2) {
      const [h, w] = dims.slice(-2);
      return {
        x: Math.min(Math.sqrt(w) * 0.1, 2) * baseScale,
        y: Math.min(Math.sqrt(h) * 0.1, 2) * baseScale,
        z: 0.3 * baseScale,
      };
    }
  }
  
  return { x: baseScale, y: baseScale, z: 0.3 * baseScale };
}

// Selectors for optimized re-renders
export const selectModel = (state: VisualizerState) => state.model;
export const selectComputedNodes = (state: VisualizerState) => state.computedNodes;
export const selectComputedEdges = (state: VisualizerState) => state.computedEdges;
export const selectConfig = (state: VisualizerState) => state.config;
export const selectSelection = (state: VisualizerState) => state.selection;
export const selectCamera = (state: VisualizerState) => state.camera;
export const selectIsLoading = (state: VisualizerState) => state.isLoading;
export const selectError = (state: VisualizerState) => state.error;
