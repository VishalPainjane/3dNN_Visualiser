import React, { useCallback, useState } from 'react';
import { useVisualizerStore } from '@/core/store';
import { loadModelFromFile, SUPPORTED_EXTENSIONS } from '@/core/loader';
import { computeLayout } from '@/core/layout';
import { isSupportedExtension, getFormatDisplayName, detectFormatFromExtension } from '@/core/formats';
import { SavedModelsPanel } from '@/components/visualization/SavedModelsPanel';
import styles from './DropZone.module.css';

// Example CNN model for demo purposes
const EXAMPLE_MODEL = {
  name: 'EXAMPLE_CNN',
  framework: 'demo',
  totalParameters: 1234567,
  trainableParameters: 1234567,
  inputShape: [1, 3, 224, 224],
  outputShape: [1, 1000],
  layers: [
    { id: 'input', name: 'input', type: 'Input', category: 'input', inputShape: null, outputShape: [1, 3, 224, 224], params: {}, numParameters: 0, trainable: false },
    { id: 'conv1', name: 'conv1', type: 'Conv2d', category: 'convolution', inputShape: [1, 3, 224, 224], outputShape: [1, 64, 112, 112], params: { filters: 64, kernelSize: [7, 7], strides: [2, 2], padding: 'same' }, numParameters: 9472, trainable: true },
    { id: 'bn1', name: 'bn1', type: 'BatchNorm2d', category: 'normalization', inputShape: [1, 64, 112, 112], outputShape: [1, 64, 112, 112], params: { epsilon: 0.00001, momentum: 0.1 }, numParameters: 256, trainable: true },
    { id: 'relu1', name: 'relu1', type: 'ReLU', category: 'activation', inputShape: [1, 64, 112, 112], outputShape: [1, 64, 112, 112], params: {}, numParameters: 0, trainable: false },
    { id: 'pool1', name: 'pool1', type: 'MaxPool2d', category: 'pooling', inputShape: [1, 64, 112, 112], outputShape: [1, 64, 56, 56], params: { kernelSize: [3, 3], strides: [2, 2] }, numParameters: 0, trainable: false },
    { id: 'conv2', name: 'conv2', type: 'Conv2d', category: 'convolution', inputShape: [1, 64, 56, 56], outputShape: [1, 128, 56, 56], params: { filters: 128, kernelSize: [3, 3], strides: [1, 1], padding: 'same' }, numParameters: 73856, trainable: true },
    { id: 'bn2', name: 'bn2', type: 'BatchNorm2d', category: 'normalization', inputShape: [1, 128, 56, 56], outputShape: [1, 128, 56, 56], params: { epsilon: 0.00001, momentum: 0.1 }, numParameters: 512, trainable: true },
    { id: 'relu2', name: 'relu2', type: 'ReLU', category: 'activation', inputShape: [1, 128, 56, 56], outputShape: [1, 128, 56, 56], params: {}, numParameters: 0, trainable: false },
    { id: 'conv3', name: 'conv3', type: 'Conv2d', category: 'convolution', inputShape: [1, 128, 56, 56], outputShape: [1, 256, 28, 28], params: { filters: 256, kernelSize: [3, 3], strides: [2, 2], padding: 'same' }, numParameters: 295168, trainable: true },
    { id: 'bn3', name: 'bn3', type: 'BatchNorm2d', category: 'normalization', inputShape: [1, 256, 28, 28], outputShape: [1, 256, 28, 28], params: { epsilon: 0.00001, momentum: 0.1 }, numParameters: 1024, trainable: true },
    { id: 'relu3', name: 'relu3', type: 'ReLU', category: 'activation', inputShape: [1, 256, 28, 28], outputShape: [1, 256, 28, 28], params: {}, numParameters: 0, trainable: false },
    { id: 'conv4', name: 'conv4', type: 'Conv2d', category: 'convolution', inputShape: [1, 256, 28, 28], outputShape: [1, 512, 14, 14], params: { filters: 512, kernelSize: [3, 3], strides: [2, 2], padding: 'same' }, numParameters: 1180160, trainable: true },
    { id: 'bn4', name: 'bn4', type: 'BatchNorm2d', category: 'normalization', inputShape: [1, 512, 14, 14], outputShape: [1, 512, 14, 14], params: { epsilon: 0.00001, momentum: 0.1 }, numParameters: 2048, trainable: true },
    { id: 'relu4', name: 'relu4', type: 'ReLU', category: 'activation', inputShape: [1, 512, 14, 14], outputShape: [1, 512, 14, 14], params: {}, numParameters: 0, trainable: false },
    { id: 'gap', name: 'global_avg_pool', type: 'AdaptiveAvgPool2d', category: 'pooling', inputShape: [1, 512, 14, 14], outputShape: [1, 512, 1, 1], params: { outputSize: [1, 1] }, numParameters: 0, trainable: false },
    { id: 'flatten', name: 'flatten', type: 'Flatten', category: 'reshape', inputShape: [1, 512, 1, 1], outputShape: [1, 512], params: {}, numParameters: 0, trainable: false },
    { id: 'fc', name: 'fc', type: 'Linear', category: 'linear', inputShape: [1, 512], outputShape: [1, 1000], params: { inFeatures: 512, outFeatures: 1000, bias: true }, numParameters: 513000, trainable: true },
    { id: 'softmax', name: 'softmax', type: 'Softmax', category: 'output', inputShape: [1, 1000], outputShape: [1, 1000], params: { dim: 1 }, numParameters: 0, trainable: false },
  ],
  connections: [
    { source: 'input', target: 'conv1', tensorShape: [1, 3, 224, 224] },
    { source: 'conv1', target: 'bn1', tensorShape: [1, 64, 112, 112] },
    { source: 'bn1', target: 'relu1', tensorShape: [1, 64, 112, 112] },
    { source: 'relu1', target: 'pool1', tensorShape: [1, 64, 112, 112] },
    { source: 'pool1', target: 'conv2', tensorShape: [1, 64, 56, 56] },
    { source: 'conv2', target: 'bn2', tensorShape: [1, 128, 56, 56] },
    { source: 'bn2', target: 'relu2', tensorShape: [1, 128, 56, 56] },
    { source: 'relu2', target: 'conv3', tensorShape: [1, 128, 56, 56] },
    { source: 'conv3', target: 'bn3', tensorShape: [1, 256, 28, 28] },
    { source: 'bn3', target: 'relu3', tensorShape: [1, 256, 28, 28] },
    { source: 'relu3', target: 'conv4', tensorShape: [1, 256, 28, 28] },
    { source: 'conv4', target: 'bn4', tensorShape: [1, 512, 14, 14] },
    { source: 'bn4', target: 'relu4', tensorShape: [1, 512, 14, 14] },
    { source: 'relu4', target: 'gap', tensorShape: [1, 512, 14, 14] },
    { source: 'gap', target: 'flatten', tensorShape: [1, 512, 1, 1] },
    { source: 'flatten', target: 'fc', tensorShape: [1, 512] },
    { source: 'fc', target: 'softmax', tensorShape: [1, 1000] },
  ],
};

interface DropZoneProps {
  children?: React.ReactNode;
}

/**
 * File drop zone overlay when no model is loaded
 */
export function DropZone({ children }: DropZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('');
  const [showSavedModels, setShowSavedModels] = useState(false);
  const model = useVisualizerStore(state => state.model);
  const loadModel = useVisualizerStore(state => state.loadModel);
  const updateNodePositions = useVisualizerStore(state => state.updateNodePositions);
  const config = useVisualizerStore(state => state.config);
  
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);
  
  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);
  
  const processFile = useCallback(async (file: File) => {
    // Check file extension
    if (!isSupportedExtension(file.name)) {
      const ext = '.' + file.name.split('.').pop()?.toLowerCase();
      alert(
        `Unsupported file format: ${ext}\n\n` +
        `Supported formats:\n` +
        `• .nn3d, .json - Native format\n` +
        `• .onnx - ONNX models\n` +
        `• .safetensors - SafeTensors (LLM weights)\n` +
        `• .pt, .pth, .ckpt - PyTorch (requires conversion)\n` +
        `• .h5, .hdf5 - Keras/TensorFlow (requires conversion)`
      );
      return;
    }
    
    const formatInfo = detectFormatFromExtension(file.name);
    const formatName = getFormatDisplayName(formatInfo.category);
    
    setIsLoading(true);
    setLoadingMessage(`Loading ${formatName} model...`);
    
    try {
      const loadedModel = await loadModelFromFile(file);
      loadModel(loadedModel);
      
      setLoadingMessage('Computing layout...');
      
      const layoutResult = computeLayout(loadedModel, {
        type: config.layout || 'layered',
        layerSpacing: config.layerSpacing || 3,
      });
      updateNodePositions(layoutResult.positions);
      
    } catch (error) {
      console.error('Failed to load model:', error);
      alert(`Failed to load model:\n\n${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsLoading(false);
      setLoadingMessage('');
    }
  }, [loadModel, updateNodePositions, config]);
  
  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    if (files.length === 0) return;
    
    await processFile(files[0]);
  }, [processFile]);
  
  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      processFile(files[0]);
    }
  }, [processFile]);

  // Handle loading a saved model from the panel
  const handleLoadSavedModel = useCallback((architecture: any) => {
    // Create a model structure from the saved architecture (matching App.tsx format)
    const savedModel = {
      version: '1.0',
      metadata: {
        name: architecture.name,
        framework: architecture.framework,
        totalParams: architecture.totalParameters,
        trainableParams: architecture.trainableParameters,
        inputShape: architecture.inputShape,
        outputShape: architecture.outputShape,
      },
      graph: {
        nodes: architecture.layers.map((layer: any) => ({
          id: layer.id,
          name: layer.name,
          type: layer.type,
          inputShape: layer.inputShape,
          outputShape: layer.outputShape,
          params: layer.params || {},
          attributes: {
            category: layer.category,
            parameters: layer.numParameters,
          },
        })),
        edges: architecture.connections?.map((conn: any, idx: number) => ({
          id: `edge-${idx}`,
          source: conn.source,
          target: conn.target,
          tensorShape: conn.tensorShape,
        })) || [],
      },
    };
    
    loadModel(savedModel as any);
    
    // Compute layout for the loaded model
    const layoutResult = computeLayout(savedModel as any, {
      type: config.layout || 'layered',
      layerSpacing: config.layerSpacing || 3,
    });
    updateNodePositions(layoutResult.positions);
  }, [loadModel, updateNodePositions, config]);

  // Handle loading the example model
  const handleLoadExample = useCallback(() => {
    handleLoadSavedModel(EXAMPLE_MODEL);
  }, [handleLoadSavedModel]);
  
  return (
    <div
      className={styles.container}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {children}
      
      {/* Loading overlay */}
      {isLoading && (
        <div className={styles.loadingOverlay}>
          <div className={styles.spinner}></div>
          <p>{loadingMessage}</p>
        </div>
      )}
      
      {!model && !isLoading && (
        <div className={`${styles.overlay} ${isDragging ? styles.dragging : ''}`}>
          <div className={styles.content}>
            <div className={styles.icon}>[NN]</div>
            <h2>NN3D_VISUALIZER</h2>
            <p>// DROP A MODEL FILE TO VISUALIZE</p>
            
            <div className={styles.formats}>
              <span className={styles.formatBadge} data-supported="true">.nn3d</span>
              <span className={styles.formatBadge} data-supported="true">.onnx</span>
              <span className={styles.formatBadge} data-supported="true">.safetensors</span>
              <span className={styles.formatBadge} data-supported="partial">.pt/.pth</span>
              <span className={styles.formatBadge} data-supported="partial">.h5</span>
            </div>
            
            <label className={styles.uploadButton}>
              <input 
                type="file" 
                accept={SUPPORTED_EXTENSIONS.join(',')}
                onChange={handleFileInput}
                style={{ display: 'none' }}
              />
              [UPLOAD_FILE]
            </label>
            
            <p className={styles.hint}>
              SUPPORTS: ONNX | SAFETENSORS | PYTORCH | KERAS
            </p>
            
            <div className={styles.features}>
              <span>[*] 3D_VISUALIZATION</span>
              <span>[*] INTERACTIVE_EXPLORATION</span>
              <span>[*] LAYER_DETAILS</span>
            </div>

            <div className={styles.divider}></div>

            <div className={styles.actionButtons}>
              <button 
                className={styles.exampleButton}
                onClick={handleLoadExample}
              >
                [DEMO] TRY_EXAMPLE_MODEL
              </button>
              
              <button 
                className={styles.savedModelsButton}
                onClick={() => setShowSavedModels(true)}
              >
                [&gt;_] BROWSE_SAVED_MODELS
              </button>
            </div>
          </div>
        </div>
      )}
      
      {isDragging && model && (
        <div className={styles.dropIndicator}>
          <span>Drop to load new model</span>
        </div>
      )}

      {/* Saved Models Panel */}
      {showSavedModels && (
        <SavedModelsPanel
          onLoadModel={handleLoadSavedModel}
          onClose={() => setShowSavedModels(false)}
        />
      )}
    </div>
  );
}

export default DropZone;
