import { useMemo, useRef } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import { Line, QuadraticBezierLine } from '@react-three/drei';
import type { ComputedEdge } from '@/core/store';
import type { Position3D } from '@/schema/types';

/**
 * Edge style type
 */
export type EdgeStyle = 'line' | 'tube' | 'arrow' | 'bezier';

/**
 * Props for edge components
 */
export interface EdgeProps {
  edge: ComputedEdge;
  style?: EdgeStyle;
  animated?: boolean;
  flowSpeed?: number;
}

/**
 * Calculate control points for bezier curves
 */
function getBezierControlPoints(
  start: Position3D,
  end: Position3D
): { mid1: Position3D; mid2: Position3D } {
  const midY = (start.y + end.y) / 2;
  const offsetZ = Math.abs(end.y - start.y) * 0.3;
  
  return {
    mid1: { x: start.x, y: midY, z: start.z + offsetZ },
    mid2: { x: end.x, y: midY, z: end.z + offsetZ },
  };
}

/**
 * Simple line edge
 */
export function LineEdge({ edge }: EdgeProps) {
  const points = useMemo(() => [
    new THREE.Vector3(edge.sourcePosition.x, edge.sourcePosition.y, edge.sourcePosition.z),
    new THREE.Vector3(edge.targetPosition.x, edge.targetPosition.y, edge.targetPosition.z),
  ], [edge.sourcePosition, edge.targetPosition]);
  
  const color = edge.highlighted ? '#ffffff' : edge.color;
  const lineWidth = edge.highlighted ? 3 : 1.5;
  
  return (
    <Line
      points={points}
      color={color}
      lineWidth={lineWidth}
      opacity={edge.visible ? 1 : 0.2}
      transparent
    />
  );
}

/**
 * Bezier curve edge for smoother connections
 */
export function BezierEdge({ edge }: EdgeProps) {
  const start = useMemo(() => 
    new THREE.Vector3(edge.sourcePosition.x, edge.sourcePosition.y, edge.sourcePosition.z),
    [edge.sourcePosition]
  );
  
  const end = useMemo(() =>
    new THREE.Vector3(edge.targetPosition.x, edge.targetPosition.y, edge.targetPosition.z),
    [edge.targetPosition]
  );
  
  const control = useMemo(() => {
    const { mid1 } = getBezierControlPoints(edge.sourcePosition, edge.targetPosition);
    return new THREE.Vector3(mid1.x, mid1.y, mid1.z);
  }, [edge.sourcePosition, edge.targetPosition]);
  
  const color = edge.highlighted ? '#ffffff' : edge.color;
  const lineWidth = edge.highlighted ? 3 : 1.5;
  
  return (
    <QuadraticBezierLine
      start={start}
      end={end}
      mid={control}
      color={color}
      lineWidth={lineWidth}
      opacity={edge.visible ? 1 : 0.2}
      transparent
    />
  );
}

/**
 * Tube edge for 3D pipe-like connections
 */
export function TubeEdge({ edge, animated = false, flowSpeed = 1 }: EdgeProps) {
  const tubeRef = useRef<THREE.Mesh>(null);
  
  // Create curve for tube
  const curve = useMemo(() => {
    const start = new THREE.Vector3(
      edge.sourcePosition.x,
      edge.sourcePosition.y,
      edge.sourcePosition.z
    );
    const end = new THREE.Vector3(
      edge.targetPosition.x,
      edge.targetPosition.y,
      edge.targetPosition.z
    );
    
    const { mid1, mid2 } = getBezierControlPoints(edge.sourcePosition, edge.targetPosition);
    const control1 = new THREE.Vector3(mid1.x, mid1.y, mid1.z);
    const control2 = new THREE.Vector3(mid2.x, mid2.y, mid2.z);
    
    return new THREE.CubicBezierCurve3(start, control1, control2, end);
  }, [edge.sourcePosition, edge.targetPosition]);
  
  // Tube geometry
  const geometry = useMemo(() => {
    const radius = edge.highlighted ? 0.06 : 0.03;
    return new THREE.TubeGeometry(curve, 32, radius, 8, false);
  }, [curve, edge.highlighted]);
  
  // Animate flow effect
  useFrame(({ clock }) => {
    if (animated && tubeRef.current) {
      const material = tubeRef.current.material as THREE.MeshStandardMaterial;
      if (material.map) {
        material.map.offset.y = clock.getElapsedTime() * flowSpeed;
      }
    }
  });
  
  const color = edge.highlighted ? '#ffffff' : edge.color;
  
  return (
    <mesh ref={tubeRef} geometry={geometry}>
      <meshStandardMaterial
        color={color}
        opacity={edge.visible ? 0.8 : 0.2}
        transparent
        metalness={0.3}
        roughness={0.6}
      />
    </mesh>
  );
}

/**
 * Arrow edge with direction indicator
 */
export function ArrowEdge({ edge }: EdgeProps) {
  const start = new THREE.Vector3(
    edge.sourcePosition.x,
    edge.sourcePosition.y,
    edge.sourcePosition.z
  );
  const end = new THREE.Vector3(
    edge.targetPosition.x,
    edge.targetPosition.y,
    edge.targetPosition.z
  );
  
  const direction = end.clone().sub(start).normalize();
  const arrowPosition = end.clone().sub(direction.clone().multiplyScalar(0.3));
  
  const color = edge.highlighted ? '#ffffff' : edge.color;
  
  // Calculate arrow rotation
  const quaternion = new THREE.Quaternion();
  quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
  
  return (
    <group>
      <LineEdge edge={edge} />
      {/* Arrow head */}
      <mesh position={arrowPosition} quaternion={quaternion}>
        <coneGeometry args={[0.1, 0.2, 8]} />
        <meshStandardMaterial color={color} />
      </mesh>
    </group>
  );
}

/**
 * Factory function to get edge component by style
 */
export function getEdgeComponent(style: EdgeStyle): React.ComponentType<EdgeProps> {
  switch (style) {
    case 'line':
      return LineEdge;
    case 'bezier':
      return BezierEdge;
    case 'tube':
      return TubeEdge;
    case 'arrow':
      return ArrowEdge;
    default:
      return TubeEdge;
  }
}
