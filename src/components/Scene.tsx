import { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { Stars, GizmoHelper, GizmoViewport } from '@react-three/drei';
import { LayerNodes } from './layers';
import { EdgeConnections } from './edges';
import { CameraControls, useKeyboardShortcuts } from './controls';
import { useVisualizerStore } from '@/core/store';

/**
 * Loading fallback component
 */
function LoadingFallback() {
  return (
    <mesh>
      <sphereGeometry args={[0.5, 32, 32]} />
      <meshStandardMaterial color="#4fc3f7" wireframe />
    </mesh>
  );
}

/**
 * Scene lighting setup
 */
function SceneLighting() {
  const config = useVisualizerStore(state => state.config);
  const isDark = config.theme !== 'light';
  
  return (
    <>
      <ambientLight intensity={isDark ? 0.4 : 0.6} />
      <directionalLight
        position={[10, 20, 10]}
        intensity={isDark ? 0.8 : 1}
        castShadow
        shadow-mapSize={[2048, 2048]}
      />
      <directionalLight position={[-10, 10, -10]} intensity={0.3} />
      <pointLight position={[0, 10, 0]} intensity={0.5} color="#4fc3f7" />
    </>
  );
}

/**
 * Scene background
 */
function SceneBackground() {
  const config = useVisualizerStore(state => state.config);
  
  if (config.theme === 'light') {
    return <color attach="background" args={['#f0f0f0']} />;
  }
  
  if (config.theme === 'blueprint') {
    return (
      <>
        <color attach="background" args={['#0a1929']} />
        <gridHelper args={[100, 100, '#1e3a5f', '#0d2137']} position={[0, -10, 0]} />
      </>
    );
  }
  
  // Dark theme (default)
  return (
    <>
      <color attach="background" args={['#0f0f1a']} />
      <Stars radius={100} depth={50} count={2000} factor={4} fade speed={0.5} />
    </>
  );
}

/**
 * Keyboard shortcuts handler component
 */
function KeyboardHandler() {
  useKeyboardShortcuts();
  return null;
}

/**
 * Grid and helper elements
 */
function SceneHelpers() {
  const model = useVisualizerStore(state => state.model);
  
  if (!model) return null;
  
  return (
    <>
      <gridHelper
        args={[50, 50, '#333', '#222']}
        position={[0, -15, 0]}
        rotation={[0, 0, 0]}
      />
      <GizmoHelper alignment="bottom-left" margin={[80, 80]}>
        <GizmoViewport axisColors={['#f44336', '#4caf50', '#2196f3']} labelColor="white" />
      </GizmoHelper>
    </>
  );
}

/**
 * Main 3D network scene
 */
function NetworkScene() {
  return (
    <>
      <SceneLighting />
      <SceneBackground />
      <SceneHelpers />
      
      <Suspense fallback={<LoadingFallback />}>
        <EdgeConnections />
        <LayerNodes />
      </Suspense>
    </>
  );
}

/**
 * Main 3D Canvas component
 */
export function Scene() {
  return (
    <Canvas
      camera={{
        position: [0, 5, 20],
        fov: 60,
        near: 0.1,
        far: 1000,
      }}
      dpr={[1, 2]}
      gl={{
        antialias: true,
        alpha: false,
        powerPreference: 'high-performance',
      }}
    >
      <CameraControls />
      <KeyboardHandler />
      <NetworkScene />
    </Canvas>
  );
}

export default Scene;
