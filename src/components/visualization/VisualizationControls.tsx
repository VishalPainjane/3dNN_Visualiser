/**
 * Visualization Controls
 * 
 * UI controls for the 3D neural network visualization.
 */

import React from 'react';
import styles from './VisualizationControls.module.css';

export interface VisualizationControlsProps {
  // View level
  level: 1 | 2 | 3;
  onLevelChange: (level: 1 | 2 | 3) => void;
  
  // Display options
  showLabels: boolean;
  onShowLabelsChange: (show: boolean) => void;
  
  showConnections: boolean;
  onShowConnectionsChange: (show: boolean) => void;
  
  animateFlow: boolean;
  onAnimateFlowChange: (animate: boolean) => void;
  
  // Model info
  modelName?: string;
  totalLayers?: number;
  totalParams?: number;
  
  // Actions
  onResetCamera?: () => void;
  onExport?: () => void;
}

export const VisualizationControls: React.FC<VisualizationControlsProps> = ({
  level,
  onLevelChange,
  showLabels,
  onShowLabelsChange,
  showConnections,
  onShowConnectionsChange,
  animateFlow,
  onAnimateFlowChange,
  modelName,
  totalLayers,
  totalParams,
  onResetCamera,
  onExport,
}) => {
  return (
    <div className={styles.controls}>
      {/* Model Info */}
      {modelName && (
        <div className={styles.section}>
          <h3 className={styles.sectionTitle}>Model</h3>
          <div className={styles.modelInfo}>
            <span className={styles.modelName}>{modelName}</span>
            <div className={styles.stats}>
              {totalLayers && <span>{totalLayers} layers</span>}
              {totalParams && <span>{formatParams(totalParams)}</span>}
            </div>
          </div>
        </div>
      )}
      
      {/* View Level */}
      <div className={styles.section}>
        <h3 className={styles.sectionTitle}>Abstraction Level</h3>
        <div className={styles.levelButtons}>
          <button
            className={`${styles.levelButton} ${level === 1 ? styles.active : ''}`}
            onClick={() => onLevelChange(1)}
            title="Macro view: Encoder → Decoder → Head"
          >
            <span className={styles.levelIcon}>[1]</span>
            <span>MACRO</span>
          </button>
          <button
            className={`${styles.levelButton} ${level === 2 ? styles.active : ''}`}
            onClick={() => onLevelChange(2)}
            title="Stage view: Show internal blocks"
          >
            <span className={styles.levelIcon}>[2]</span>
            <span>STAGE</span>
          </button>
          <button
            className={`${styles.levelButton} ${level === 3 ? styles.active : ''}`}
            onClick={() => onLevelChange(3)}
            title="Layer view: All individual layers"
          >
            <span className={styles.levelIcon}>[3]</span>
            <span>LAYER</span>
          </button>
        </div>
      </div>
      
      {/* Display Options */}
      <div className={styles.section}>
        <h3 className={styles.sectionTitle}>Display</h3>
        <div className={styles.toggles}>
          <label className={styles.toggle}>
            <input
              type="checkbox"
              checked={showLabels}
              onChange={(e) => onShowLabelsChange(e.target.checked)}
            />
            <span className={styles.toggleLabel}>Labels</span>
          </label>
          <label className={styles.toggle}>
            <input
              type="checkbox"
              checked={showConnections}
              onChange={(e) => onShowConnectionsChange(e.target.checked)}
            />
            <span className={styles.toggleLabel}>Connections</span>
          </label>
          <label className={styles.toggle}>
            <input
              type="checkbox"
              checked={animateFlow}
              onChange={(e) => onAnimateFlowChange(e.target.checked)}
            />
            <span className={styles.toggleLabel}>Animate Flow</span>
          </label>
        </div>
      </div>
      
      {/* Actions */}
      <div className={styles.section}>
        <h3 className={styles.sectionTitle}>Actions</h3>
        <div className={styles.actions}>
          {onResetCamera && (
            <button className={styles.actionButton} onClick={onResetCamera}>
              [x] RESET_CAMERA
            </button>
          )}
          {onExport && (
            <button className={styles.actionButton} onClick={onExport}>
              [*] EXPORT_IMAGE
            </button>
          )}
        </div>
      </div>
      
      {/* Legend */}
      <div className={styles.section}>
        <h3 className={styles.sectionTitle}>Layer Types</h3>
        <div className={styles.legend}>
          <LegendItem color="#4A90D9" label="Convolution" />
          <LegendItem color="#9B59B6" label="Linear" />
          <LegendItem color="#2ECC71" label="Normalization" />
          <LegendItem color="#F39C12" label="Activation" />
          <LegendItem color="#1ABC9C" label="Pooling" />
          <LegendItem color="#E91E63" label="Attention" />
          <LegendItem color="#E74C3C" label="Recurrent" />
        </div>
      </div>
      
      {/* Keyboard Shortcuts */}
      <div className={styles.section}>
        <h3 className={styles.sectionTitle}>Shortcuts</h3>
        <div className={styles.shortcuts}>
          <div className={styles.shortcut}>
            <kbd>1</kbd><kbd>2</kbd><kbd>3</kbd>
            <span>View level</span>
          </div>
          <div className={styles.shortcut}>
            <kbd>L</kbd>
            <span>Toggle labels</span>
          </div>
          <div className={styles.shortcut}>
            <kbd>C</kbd>
            <span>Toggle connections</span>
          </div>
          <div className={styles.shortcut}>
            <kbd>Space</kbd>
            <span>Reset camera</span>
          </div>
        </div>
      </div>
    </div>
  );
};

// Helper Components

const LegendItem: React.FC<{ color: string; label: string }> = ({ color, label }) => (
  <div className={styles.legendItem}>
    <span className={styles.legendColor} style={{ backgroundColor: color }} />
    <span className={styles.legendLabel}>{label}</span>
  </div>
);

// Helper Functions

function formatParams(params: number): string {
  if (params >= 1e9) return `${(params / 1e9).toFixed(1)}B`;
  if (params >= 1e6) return `${(params / 1e6).toFixed(1)}M`;
  if (params >= 1e3) return `${(params / 1e3).toFixed(1)}K`;
  return params.toString();
}

export default VisualizationControls;
