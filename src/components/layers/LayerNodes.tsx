import { useMemo } from 'react';
import { useVisualizerStore } from '@/core/store';
import { getLayerComponent } from './LayerGeometry';
import { getLayerMeshComponent } from './NeuralLayerMesh';
import type { ComputedNode } from '@/core/store';

/**
 * Props for individual layer node component
 */
interface LayerNodeProps {
  node: ComputedNode;
}

/**
 * Single layer node in the 3D scene
 */
function LayerNode({ node }: LayerNodeProps) {
  const selectNode = useVisualizerStore(state => state.selectNode);
  const hoverNode = useVisualizerStore(state => state.hoverNode);
  const config = useVisualizerStore(state => state.config);
  
  // Check if this model was analyzed by backend (has enhanced data)
  const useEnhancedVisualization = useMemo(() => {
    // Use enhanced visualization for models with proper layer attributes
    return node.params && (
      node.params.out_features !== undefined ||
      node.params.outFeatures !== undefined ||
      node.params.out_channels !== undefined ||
      node.params.outChannels !== undefined ||
      node.params.hidden_size !== undefined ||
      node.params.hiddenSize !== undefined ||
      node.params.num_heads !== undefined ||
      node.params.numHeads !== undefined
    );
  }, [node.params]);
  
  if (!node.visible) return null;
  
  // Enhanced visualization
  if (useEnhancedVisualization) {
    const NeuralComponent = getLayerMeshComponent(node.type);
    
    return (
      <group position={[node.computedPosition.x, node.computedPosition.y, node.computedPosition.z]}>
        <NeuralComponent
          node={{
            id: node.id,
            name: node.name,
            type: node.type,
            attributes: node.params,
            computedPosition: { x: 0, y: 0, z: 0 }
          }}
          color={node.color}
          selected={node.selected}
          hovered={node.hovered}
          showNeurons={config.showLabels}
          onClick={() => selectNode(node.id)}
          onPointerOver={() => hoverNode(node.id)}
          onPointerOut={() => hoverNode(null)}
        />
      </group>
    );
  }
  
  // Original visualization
  const LayerComponent = getLayerComponent(node.type);
  
  return (
    <group position={[node.computedPosition.x, node.computedPosition.y, node.computedPosition.z]}>
      <LayerComponent
        type={node.type}
        params={node.params}
        inputShape={node.inputShape}
        outputShape={node.outputShape}
        color={node.color}
        scale={node.scale}
        selected={node.selected}
        hovered={node.hovered}
        lod={node.lod}
        showLabel={config.showLabels}
        label={node.name}
        onClick={() => selectNode(node.id)}
        onPointerOver={() => hoverNode(node.id)}
        onPointerOut={() => hoverNode(null)}
      />
    </group>
  );
}

/**
 * Renders all layer nodes in the network
 */
export function LayerNodes() {
  const computedNodes = useVisualizerStore(state => state.computedNodes);
  
  const nodeArray = useMemo(() => Array.from(computedNodes.values()), [computedNodes]);
  
  return (
    <group name="layer-nodes">
      {nodeArray.map(node => (
        <LayerNode key={node.id} node={node} />
      ))}
    </group>
  );
}

export { LayerNode };
