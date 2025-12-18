/**
 * Layer Detail Panel
 * 
 * Shows detailed information about the selected layer.
 */

import React from 'react';
import type { HierarchyNode } from '@/core/model-hierarchy';
import { getLayerColor } from '@/core/layer-geometry';
import styles from './LayerDetailPanel.module.css';

export interface LayerDetailPanelProps {
  node: HierarchyNode | null;
  onClose: () => void;
}

export const LayerDetailPanel: React.FC<LayerDetailPanelProps> = ({
  node,
  onClose,
}) => {
  if (!node) return null;
  
  const color = getLayerColor(node.category);
  const layer = node.layerData;
  
  return (
    <div className={styles.panel}>
      <div className={styles.header} style={{ borderLeftColor: color }}>
        <div className={styles.headerContent}>
          <span 
            className={styles.categoryBadge}
            style={{ backgroundColor: color }}
          >
            {node.category}
          </span>
          <h3 className={styles.title}>{node.displayName}</h3>
          <p className={styles.fullName}>{node.name}</p>
        </div>
        <button className={styles.closeButton} onClick={onClose}>
          ×
        </button>
      </div>
      
      <div className={styles.content}>
        {/* Type Info */}
        {layer && (
          <Section title="LAYER_TYPE">
            <div className={styles.typeInfo}>
              <span className={styles.typeName}>{layer.type}</span>
              <span className={styles.trainableBadge}>
                {layer.trainable ? '[TRAINABLE]' : '[FROZEN]'}
              </span>
            </div>
          </Section>
        )}
        
        {/* Shapes */}
        <Section title="Tensor Shapes">
          <div className={styles.shapes}>
            <ShapeRow 
              label="Input" 
              shape={node.inputShape} 
              color="#4A90D9"
            />
            <div className={styles.shapeArrow}>→</div>
            <ShapeRow 
              label="Output" 
              shape={node.outputShape} 
              color="#2ECC71"
            />
          </div>
        </Section>
        
        {/* Parameters */}
        <Section title="Parameters">
          <div className={styles.paramStats}>
            <div className={styles.paramStat}>
              <span className={styles.paramValue}>
                {formatNumber(node.totalParams)}
              </span>
              <span className={styles.paramLabel}>Total</span>
            </div>
            {node.layerCount > 1 && (
              <div className={styles.paramStat}>
                <span className={styles.paramValue}>{node.layerCount}</span>
                <span className={styles.paramLabel}>Layers</span>
              </div>
            )}
          </div>
        </Section>
        
        {/* Layer-specific params */}
        {layer?.params && Object.keys(layer.params).length > 0 && (
          <Section title="Configuration">
            <div className={styles.configTable}>
              {Object.entries(layer.params).map(([key, value]) => (
                <div key={key} className={styles.configRow}>
                  <span className={styles.configKey}>{formatKey(key)}</span>
                  <span className={styles.configValue}>{formatValue(value)}</span>
                </div>
              ))}
            </div>
          </Section>
        )}
        
        {/* Children (for groups) */}
        {node.children.length > 0 && (
          <Section title={`Contains (${node.children.length})`}>
            <div className={styles.childList}>
              {node.children.slice(0, 10).map(child => (
                <div key={child.id} className={styles.childItem}>
                  <span 
                    className={styles.childDot}
                    style={{ backgroundColor: getLayerColor(child.category) }}
                  />
                  <span className={styles.childName}>{child.displayName}</span>
                  <span className={styles.childType}>{child.layerData?.type || child.category}</span>
                </div>
              ))}
              {node.children.length > 10 && (
                <div className={styles.moreChildren}>
                  + {node.children.length - 10} more...
                </div>
              )}
            </div>
          </Section>
        )}
        
        {/* Position Info */}
        <Section title="Hierarchy">
          <div className={styles.hierarchyInfo}>
            <div className={styles.hierarchyRow}>
              <span>Level</span>
              <span>{getLevelName(node.level)}</span>
            </div>
            <div className={styles.hierarchyRow}>
              <span>Depth</span>
              <span>{node.depth.toFixed(1)}</span>
            </div>
            <div className={styles.hierarchyRow}>
              <span>Channels</span>
              <span>{node.channelSize}</span>
            </div>
          </div>
        </Section>
      </div>
    </div>
  );
};

// Helper Components

const Section: React.FC<{ title: string; children: React.ReactNode }> = ({ 
  title, 
  children 
}) => (
  <div className={styles.section}>
    <h4 className={styles.sectionTitle}>{title}</h4>
    {children}
  </div>
);

const ShapeRow: React.FC<{ 
  label: string; 
  shape: number[] | null;
  color: string;
}> = ({ label, shape, color }) => (
  <div className={styles.shapeRow}>
    <span className={styles.shapeLabel} style={{ color }}>{label}</span>
    <code className={styles.shapeValue}>
      {shape ? `[${shape.map(d => d === -1 ? '?' : d).join(', ')}]` : 'N/A'}
    </code>
  </div>
);

// Helper Functions

function formatNumber(num: number): string {
  if (num >= 1e9) return `${(num / 1e9).toFixed(2)}B`;
  if (num >= 1e6) return `${(num / 1e6).toFixed(2)}M`;
  if (num >= 1e3) return `${(num / 1e3).toFixed(2)}K`;
  return num.toLocaleString();
}

function formatKey(key: string): string {
  return key
    .replace(/_/g, ' ')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return 'N/A';
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (Array.isArray(value)) return `[${value.join(', ')}]`;
  if (typeof value === 'number') {
    return Number.isInteger(value) ? value.toString() : value.toFixed(4);
  }
  return String(value);
}

function getLevelName(level: 1 | 2 | 3): string {
  switch (level) {
    case 1: return 'Macro (Module)';
    case 2: return 'Stage (Block)';
    case 3: return 'Layer (Operation)';
  }
}

export default LayerDetailPanel;
