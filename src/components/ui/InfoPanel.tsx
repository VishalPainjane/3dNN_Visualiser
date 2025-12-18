import { useVisualizerStore } from '@/core/store';
import type { LayerType, LayerParams, TensorShape } from '@/schema/types';
import { LAYER_CATEGORIES } from '@/schema/types';
import styles from './InfoPanel.module.css';

/**
 * Format tensor shape for display
 */
function formatShape(shape?: TensorShape): string {
  if (!shape || shape.length === 0) return 'N/A';
  return `[${shape.join(', ')}]`;
}

/**
 * Format parameter value for display
 */
function formatParamValue(value: unknown): string {
  if (value === null || value === undefined) return 'N/A';
  if (Array.isArray(value)) return `[${value.join(', ')}]`;
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (typeof value === 'number') return value.toLocaleString();
  return String(value);
}

/**
 * Get human-readable layer type name
 */
function getLayerTypeName(type: LayerType): string {
  const names: Partial<Record<LayerType, string>> = {
    conv2d: '2D Convolution',
    conv3d: '3D Convolution',
    convTranspose2d: 'Transposed Conv2D',
    linear: 'Linear (Dense)',
    batchNorm2d: 'Batch Normalization 2D',
    layerNorm: 'Layer Normalization',
    multiHeadAttention: 'Multi-Head Attention',
    maxPool2d: 'Max Pooling 2D',
    avgPool2d: 'Average Pooling 2D',
    globalAvgPool: 'Global Average Pooling',
    relu: 'ReLU Activation',
    gelu: 'GELU Activation',
    sigmoid: 'Sigmoid Activation',
    softmax: 'Softmax',
  };
  return names[type] || type.charAt(0).toUpperCase() + type.slice(1);
}

/**
 * Layer parameter display component
 */
function LayerParamsDisplay({ params }: { params?: LayerParams }) {
  if (!params) return null;
  
  const relevantParams = Object.entries(params).filter(([_, v]) => v !== undefined && v !== null);
  
  if (relevantParams.length === 0) return null;
  
  return (
    <div className={styles.section}>
      <h4>Parameters</h4>
      <table className={styles.paramsTable}>
        <tbody>
          {relevantParams.map(([key, value]) => (
            <tr key={key}>
              <td className={styles.paramKey}>{key}</td>
              <td className={styles.paramValue}>{formatParamValue(value)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * Info panel for selected node
 */
export function InfoPanel() {
  const selection = useVisualizerStore(state => state.selection);
  const computedNodes = useVisualizerStore(state => state.computedNodes);
  const selectNode = useVisualizerStore(state => state.selectNode);
  
  const selectedNode = selection.selectedNodeId
    ? computedNodes.get(selection.selectedNodeId)
    : null;
  
  if (!selectedNode) {
    return (
      <div className={styles.panel}>
        <div className={styles.empty}>
          <p>Click on a layer to view details</p>
          <p className={styles.hint}>Use mouse to orbit, scroll to zoom</p>
        </div>
      </div>
    );
  }
  
  const category = LAYER_CATEGORIES[selectedNode.type];
  
  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <h3>{selectedNode.name}</h3>
        <button className={styles.closeButton} onClick={() => selectNode(null)}>
          ×
        </button>
      </div>
      
      <div className={styles.content}>
        <div className={styles.section}>
          <div className={styles.layerType}>
            <span
              className={styles.categoryBadge}
              style={{ backgroundColor: selectedNode.color }}
            >
              {category}
            </span>
            <span className={styles.typeName}>
              {getLayerTypeName(selectedNode.type)}
            </span>
          </div>
        </div>
        
        <div className={styles.section}>
          <h4>Shapes</h4>
          <div className={styles.shapeRow}>
            <span className={styles.shapeLabel}>Input:</span>
            <code className={styles.shapeValue}>
              {formatShape(selectedNode.inputShape)}
            </code>
          </div>
          <div className={styles.shapeRow}>
            <span className={styles.shapeLabel}>Output:</span>
            <code className={styles.shapeValue}>
              {formatShape(selectedNode.outputShape)}
            </code>
          </div>
        </div>
        
        <LayerParamsDisplay params={selectedNode.params} />
        
        {selectedNode.attributes && Object.keys(selectedNode.attributes).length > 0 && (
          <div className={styles.section}>
            <h4>Attributes</h4>
            <pre className={styles.attributes}>
              {JSON.stringify(selectedNode.attributes, null, 2)}
            </pre>
          </div>
        )}
        
        <div className={styles.section}>
          <h4>Position</h4>
          <code className={styles.position}>
            ({selectedNode.computedPosition.x.toFixed(2)},{' '}
            {selectedNode.computedPosition.y.toFixed(2)},{' '}
            {selectedNode.computedPosition.z.toFixed(2)})
          </code>
        </div>
      </div>
    </div>
  );
}

export default InfoPanel;
