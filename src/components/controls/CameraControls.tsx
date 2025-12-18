import { useRef, useEffect } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import { OrbitControls as DreiOrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { useVisualizerStore } from '@/core/store';

/**
 * Camera controls component with orbit, zoom, and pan
 */
export function CameraControls() {
  const controlsRef = useRef<any>(null);
  const model = useVisualizerStore(state => state.model);
  const setCameraPosition = useVisualizerStore(state => state.setCameraPosition);
  const setCameraTarget = useVisualizerStore(state => state.setCameraTarget);
  
  const { camera: threeCamera } = useThree();
  
  // Update store when camera moves
  useFrame(() => {
    if (controlsRef.current) {
      const pos = threeCamera.position;
      const target = controlsRef.current.target;
      
      // Debounce updates
      setCameraPosition({ x: pos.x, y: pos.y, z: pos.z });
      setCameraTarget({ x: target.x, y: target.y, z: target.z });
    }
  });
  
  // Reset camera when model changes
  useEffect(() => {
    if (model && controlsRef.current) {
      // Compute camera position to frame the model
      const nodeCount = model.graph.nodes.length;
      const distance = Math.max(nodeCount * 1.5, 15);
      
      threeCamera.position.set(0, distance * 0.3, distance);
      controlsRef.current.target.set(0, -nodeCount * 0.5, 0);
      controlsRef.current.update();
    }
  }, [model, threeCamera]);
  
  return (
    <DreiOrbitControls
      ref={controlsRef}
      enablePan={true}
      enableZoom={true}
      enableRotate={true}
      minDistance={2}
      maxDistance={100}
      minPolarAngle={0}
      maxPolarAngle={Math.PI}
      dampingFactor={0.1}
      rotateSpeed={0.5}
      panSpeed={0.5}
      zoomSpeed={0.8}
    />
  );
}

/**
 * Camera animation for transitions
 */
export function useCameraAnimation() {
  const { camera } = useThree();
  const targetRef = useRef<THREE.Vector3 | null>(null);
  const lookAtRef = useRef<THREE.Vector3 | null>(null);
  const progressRef = useRef(0);
  
  useFrame((_, delta) => {
    if (targetRef.current && progressRef.current < 1) {
      progressRef.current += delta * 2;
      const t = Math.min(progressRef.current, 1);
      const eased = 1 - Math.pow(1 - t, 3); // Ease out cubic
      
      camera.position.lerp(targetRef.current, eased);
      
      if (lookAtRef.current) {
        camera.lookAt(lookAtRef.current);
      }
      
      if (t >= 1) {
        targetRef.current = null;
        lookAtRef.current = null;
      }
    }
  });
  
  const animateTo = (position: THREE.Vector3, lookAt?: THREE.Vector3) => {
    targetRef.current = position;
    lookAtRef.current = lookAt || null;
    progressRef.current = 0;
  };
  
  return { animateTo };
}

/**
 * Focus camera on a specific node
 */
export function useFocusNode() {
  const computedNodes = useVisualizerStore(state => state.computedNodes);
  const { animateTo } = useCameraAnimation();
  
  const focusNode = (nodeId: string) => {
    const node = computedNodes.get(nodeId);
    if (!node) return;
    
    const { x, y, z } = node.computedPosition;
    const targetPos = new THREE.Vector3(x, y + 5, z + 10);
    const lookAtPos = new THREE.Vector3(x, y, z);
    
    animateTo(targetPos, lookAtPos);
  };
  
  return { focusNode };
}
