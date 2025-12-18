/**
 * Neural Network 3D Scene
 * 
 * Main visualization component that renders the hierarchical model.
 * Uses React Three Fiber for declarative 3D rendering.
 */

import React, { useRef, useMemo, useCallback, useState, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { 
  OrbitControls, 
  Environment,
  Text,
  Html,
  Line,
  QuadraticBezierLine,
} from '@react-three/drei';
import * as THREE from 'three';

import type { ModelHierarchy, HierarchyNode } from '@/core/model-hierarchy';
import type { LayoutResult, NodeLayout, ConnectionLayout, Position3D } from '@/core/layout-engine';
import { computeFullLayout } from '@/core/layout-engine';
import { 
  getLayerColor, 
  getGeometryType,
} from '@/core/layer-geometry';

// ============================================================================
// Types
// ============================================================================

export interface NeuralSceneProps {
  hierarchy: ModelHierarchy | null;
  level?: 1 | 2 | 3;
  showLabels?: boolean;
  showConnections?: boolean;
  animateFlow?: boolean;
  onNodeClick?: (nodeId: string) => void;
  onNodeHover?: (nodeId: string | null) => void;
  selectedNodeId?: string | null;
  hoveredNodeId?: string | null;
}

// ============================================================================
// Layer Node Component
// ============================================================================

interface LayerNodeProps {
  node: HierarchyNode;
  layout: NodeLayout;
  isSelected: boolean;
  isHovered: boolean;
  showLabel: boolean;
  onClick: () => void;
  onPointerEnter: () => void;
  onPointerLeave: () => void;
}

const LayerNode: React.FC<LayerNodeProps> = React.memo(({
  node,
  layout,
  isSelected,
  isHovered,
  showLabel,
  onClick,
  onPointerEnter,
  onPointerLeave,
}) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hoverScale, setHoverScale] = useState(1);
  
  // Animate hover scale
  useFrame((_, delta) => {
    const targetScale = isHovered ? 1.1 : 1;
    setHoverScale(prev => THREE.MathUtils.lerp(prev, targetScale, delta * 10));
  });
  
  const color = getLayerColor(node.category);
  const geometryType = getGeometryType(node.category);
  const dims = layout.dimensions;
  
  // Create geometry based on type
  const geometry = useMemo(() => {
    switch (geometryType) {
      case 'plate':
        return new THREE.BoxGeometry(dims.depth * 0.3, dims.height, dims.width);
      case 'cylinder':
        return new THREE.CylinderGeometry(dims.width / 2.5, dims.width / 2.5, dims.height, 16);
      case 'hexagon':
        return new THREE.CylinderGeometry(dims.width / 2, dims.width / 2, dims.depth, 6);
      case 'pyramid':
        return new THREE.ConeGeometry(dims.width / 2, dims.height, 4);
      case 'container':
        return new THREE.BoxGeometry(dims.depth * 2, dims.height, dims.width * 1.5);
      case 'prism':
      case 'box':
      default:
        return new THREE.BoxGeometry(dims.depth, dims.height, dims.width);
    }
  }, [geometryType, dims]);
  
  const isContainer = node.type === 'group';
  
  return (
    <group position={[layout.position.x, layout.position.y, layout.position.z]}>
      <mesh
        ref={meshRef}
        geometry={geometry}
        scale={[hoverScale, hoverScale, hoverScale]}
        onClick={(e) => { e.stopPropagation(); onClick(); }}
        onPointerEnter={(e) => { e.stopPropagation(); onPointerEnter(); }}
        onPointerLeave={onPointerLeave}
      >
        {isContainer ? (
          <meshBasicMaterial
            color={color}
            transparent
            opacity={0.15}
            wireframe
          />
        ) : (
          <meshStandardMaterial
            color={color}
            transparent={isHovered || isSelected}
            opacity={isHovered ? 0.9 : (isSelected ? 0.95 : 1)}
            metalness={0.1}
            roughness={0.7}
            emissive={isSelected ? color : '#000000'}
            emissiveIntensity={isSelected ? 0.3 : 0}
          />
        )}
      </mesh>
      
      {/* Selection outline */}
      {isSelected && (
        <mesh geometry={geometry} scale={[1.05, 1.05, 1.05]}>
          <meshBasicMaterial color="#ffffff" wireframe />
        </mesh>
      )}
      
      {/* Label */}
      {showLabel && (
        <Text
          position={[0, dims.height / 2 + 0.3, 0]}
          fontSize={0.2}
          color="#ffffff"
          anchorX="center"
          anchorY="bottom"
          outlineWidth={0.02}
          outlineColor="#000000"
        >
          {node.displayName}
        </Text>
      )}
    </group>
  );
});

LayerNode.displayName = 'LayerNode';

// ============================================================================
// Connection Component
// ============================================================================

interface ConnectionProps {
  connection: ConnectionLayout;
  animate: boolean;
  color?: string;
}

const Connection: React.FC<ConnectionProps> = React.memo(({
  connection,
  animate: _animate, // Reserved for future animation features
  color = '#4A90D9',
}) => {
  const { sourcePosition, targetPosition, controlPoints, isSkipConnection } = connection;
  
  const lineColor = isSkipConnection ? '#00BCD4' : color;
  const opacity = isSkipConnection ? 0.6 : 0.8;
  
  // For curved connections
  if (controlPoints.length > 0) {
    const midPoint = controlPoints[0];
    return (
      <QuadraticBezierLine
        start={[sourcePosition.x, sourcePosition.y, sourcePosition.z]}
        end={[targetPosition.x, targetPosition.y, targetPosition.z]}
        mid={[midPoint.x, midPoint.y, midPoint.z]}
        color={lineColor}
        lineWidth={isSkipConnection ? 1 : 2}
        transparent
        opacity={opacity}
        dashed={isSkipConnection}
        dashScale={5}
      />
    );
  }
  
  // Straight line
  return (
    <Line
      points={[
        [sourcePosition.x, sourcePosition.y, sourcePosition.z],
        [targetPosition.x, targetPosition.y, targetPosition.z],
      ]}
      color={lineColor}
      lineWidth={2}
      transparent
      opacity={opacity}
    />
  );
});

Connection.displayName = 'Connection';

// ============================================================================
// Flow Particles (animated data flow)
// ============================================================================

interface FlowParticlesProps {
  connections: ConnectionLayout[];
  speed?: number;
}

const FlowParticles: React.FC<FlowParticlesProps> = ({ connections, speed = 1 }) => {
  const particlesRef = useRef<THREE.Points>(null);
  const progressRef = useRef<Float32Array>(new Float32Array(connections.length));
  
  useEffect(() => {
    // Initialize random progress for each particle
    progressRef.current = new Float32Array(connections.length);
    for (let i = 0; i < connections.length; i++) {
      progressRef.current[i] = Math.random();
    }
  }, [connections.length]);
  
  useFrame((_, delta) => {
    if (!particlesRef.current) return;
    
    const positions = particlesRef.current.geometry.attributes.position as THREE.BufferAttribute;
    
    for (let i = 0; i < connections.length; i++) {
      progressRef.current[i] = (progressRef.current[i] + delta * speed * 0.5) % 1;
      const t = progressRef.current[i];
      
      const conn = connections[i];
      const { sourcePosition, targetPosition } = conn;
      
      // Linear interpolation along connection
      const x = sourcePosition.x + (targetPosition.x - sourcePosition.x) * t;
      const y = sourcePosition.y + (targetPosition.y - sourcePosition.y) * t;
      const z = sourcePosition.z + (targetPosition.z - sourcePosition.z) * t;
      
      positions.setXYZ(i, x, y, z);
    }
    
    positions.needsUpdate = true;
  });
  
  const particlePositions = useMemo(() => {
    const positions = new Float32Array(connections.length * 3);
    connections.forEach((conn, i) => {
      positions[i * 3] = conn.sourcePosition.x;
      positions[i * 3 + 1] = conn.sourcePosition.y;
      positions[i * 3 + 2] = conn.sourcePosition.z;
    });
    return positions;
  }, [connections]);
  
  return (
    <points ref={particlesRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={connections.length}
          array={particlePositions}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        color="#00ff88"
        size={0.15}
        transparent
        opacity={0.8}
        sizeAttenuation
      />
    </points>
  );
};

// ============================================================================
// Camera Controller
// ============================================================================

interface CameraControllerProps {
  target: Position3D;
  initialPosition: Position3D;
}

const CameraController: React.FC<CameraControllerProps> = ({ target, initialPosition }) => {
  const { camera } = useThree();
  
  useEffect(() => {
    camera.position.set(initialPosition.x, initialPosition.y, initialPosition.z);
    camera.lookAt(target.x, target.y, target.z);
  }, [camera, target, initialPosition]);
  
  return null;
};

// ============================================================================
// Info Tooltip
// ============================================================================

interface NodeTooltipProps {
  node: HierarchyNode;
  position: Position3D;
}

const NodeTooltip: React.FC<NodeTooltipProps> = ({ node, position }) => {
  return (
    <Html
      position={[position.x, position.y + 1, position.z]}
      center
      style={{
        background: 'rgba(0, 0, 0, 0.85)',
        padding: '8px 12px',
        borderRadius: '6px',
        color: 'white',
        fontSize: '12px',
        pointerEvents: 'none',
        whiteSpace: 'nowrap',
        border: `2px solid ${getLayerColor(node.category)}`,
      }}
    >
      <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>
        {node.displayName}
      </div>
      <div style={{ opacity: 0.8 }}>
        Type: {node.layerData?.type || node.category}
      </div>
      <div style={{ opacity: 0.8 }}>
        Params: {node.totalParams.toLocaleString()}
      </div>
      {node.inputShape && (
        <div style={{ opacity: 0.8 }}>
          Shape: [{node.inputShape.join(', ')}] → [{node.outputShape?.join(', ') || '?'}]
        </div>
      )}
    </Html>
  );
};

// ============================================================================
// Main Scene Content
// ============================================================================

interface SceneContentProps extends NeuralSceneProps {
  layout: LayoutResult;
}

const SceneContent: React.FC<SceneContentProps> = ({
  hierarchy,
  layout,
  showLabels = true,
  showConnections = true,
  animateFlow = false,
  onNodeClick,
  onNodeHover,
  selectedNodeId,
  hoveredNodeId,
}) => {
  const handleNodeClick = useCallback((nodeId: string) => {
    onNodeClick?.(nodeId);
  }, [onNodeClick]);
  
  const handleNodeHover = useCallback((nodeId: string | null) => {
    onNodeHover?.(nodeId);
  }, [onNodeHover]);
  
  // Get hovered node for tooltip
  const hoveredNode = hoveredNodeId && hierarchy ? hierarchy.allNodes.get(hoveredNodeId) : null;
  const hoveredLayout = hoveredNodeId ? layout.nodes.get(hoveredNodeId) : null;
  
  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 20, 10]} intensity={1} castShadow />
      <directionalLight position={[-10, 10, -10]} intensity={0.5} />
      
      {/* Camera */}
      <CameraController
        target={layout.cameraSuggestion.target}
        initialPosition={layout.cameraSuggestion.position}
      />
      
      {/* Controls */}
      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        minDistance={5}
        maxDistance={100}
        target={[
          layout.cameraSuggestion.target.x,
          layout.cameraSuggestion.target.y,
          layout.cameraSuggestion.target.z,
        ]}
      />
      
      {/* Layer Nodes */}
      {Array.from(layout.nodes.entries()).map(([nodeId, nodeLayout]) => {
        const node = hierarchy?.allNodes.get(nodeId);
        if (!node || !nodeLayout.visible) return null;
        
        return (
          <LayerNode
            key={nodeId}
            node={node}
            layout={nodeLayout}
            isSelected={selectedNodeId === nodeId}
            isHovered={hoveredNodeId === nodeId}
            showLabel={showLabels}
            onClick={() => handleNodeClick(nodeId)}
            onPointerEnter={() => handleNodeHover(nodeId)}
            onPointerLeave={() => handleNodeHover(null)}
          />
        );
      })}
      
      {/* Connections */}
      {showConnections && layout.connections.map(conn => (
        <Connection
          key={conn.id}
          connection={conn}
          animate={animateFlow}
        />
      ))}
      
      {/* Flow Animation */}
      {animateFlow && showConnections && (
        <FlowParticles connections={layout.connections} speed={1} />
      )}
      
      {/* Hover Tooltip */}
      {hoveredNode && hoveredLayout && (
        <NodeTooltip node={hoveredNode} position={hoveredLayout.position} />
      )}
      
      {/* Environment */}
      <Environment preset="city" />
      
      {/* Ground Grid */}
      <gridHelper
        args={[100, 100, '#333333', '#222222']}
        position={[layout.bounds.center.x, layout.bounds.min.y - 1, layout.bounds.center.z]}
      />
    </>
  );
};

// ============================================================================
// Main Component
// ============================================================================

export const NeuralScene: React.FC<NeuralSceneProps> = (props) => {
  const { hierarchy, level = 3 } = props;
  
  // Compute layout
  const layout = useMemo(() => {
    if (!hierarchy) {
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
    return computeFullLayout(hierarchy, level);
  }, [hierarchy, level]);
  
  if (!hierarchy) {
    return (
      <Canvas>
        <ambientLight intensity={0.5} />
        <Text position={[0, 0, 0]} fontSize={1} color="#888888">
          Drop a model file to visualize
        </Text>
      </Canvas>
    );
  }
  
  return (
    <Canvas
      shadows
      gl={{ antialias: true, alpha: false }}
      style={{ background: '#1a1a2e' }}
    >
      <SceneContent {...props} layout={layout} />
    </Canvas>
  );
};

export default NeuralScene;
