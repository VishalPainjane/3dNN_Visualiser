/**
 * Types for model format parsing
 */

import type { NN3DModel } from '@/schema/types';

/**
 * Supported model file extensions
 */
export const SUPPORTED_EXTENSIONS = [
  '.nn3d',      // Native format
  '.json',      // JSON (may be nn3d or other)
  '.onnx',      // ONNX models
  '.pt',        // PyTorch model
  '.pth',       // PyTorch checkpoint
  '.ckpt',      // Checkpoint (multi-framework)
  '.safetensors', // SafeTensors format
  '.h5',        // Keras/TensorFlow HDF5
  '.hdf5',      // HDF5 format
  '.bin',       // Binary weights
  '.pkl',       // Pickle
  '.pickle',    // Pickle
  '.joblib',    // Joblib
  '.model',     // Generic model
  '.weights',   // Raw weights
] as const;

export type SupportedExtension = typeof SUPPORTED_EXTENSIONS[number];

/**
 * Model format categories
 */
export type FormatCategory = 
  | 'native'        // .nn3d
  | 'onnx'          // ONNX format
  | 'pytorch'       // PyTorch formats
  | 'tensorflow'    // TensorFlow/Keras formats
  | 'safetensors'   // SafeTensors
  | 'generic'       // Generic binary/pickle formats
  | 'unknown';

/**
 * Format detection result
 */
export interface FormatInfo {
  extension: string;
  category: FormatCategory;
  canParseInBrowser: boolean;
  requiresBackend: boolean;
  description: string;
  conversionHint?: string;
}

/**
 * Parse result with potential warnings
 */
export interface ParseResult {
  success: boolean;
  model?: NN3DModel;
  error?: string;
  warnings: string[];
  format: FormatInfo;
  /** If true, structure was inferred from weights only */
  inferredStructure: boolean;
}

/**
 * Parser interface for different formats
 */
export interface FormatParser {
  /** Extensions this parser handles */
  extensions: string[];
  
  /** Check if this parser can handle the file */
  canParse(file: File, buffer?: ArrayBuffer): Promise<boolean>;
  
  /** Parse the file to NN3D format */
  parse(file: File): Promise<ParseResult>;
}

/**
 * Layer info extracted from model files
 */
export interface ExtractedLayer {
  id: string;
  name: string;
  type: string;
  inputShape?: number[];
  outputShape?: number[];
  params?: Record<string, unknown>;
  attributes?: Record<string, unknown>;
}

/**
 * Connection info extracted from model files  
 */
export interface ExtractedConnection {
  source: string;
  target: string;
  sourceOutput?: number;
  targetInput?: number;
}

/**
 * Raw extracted model structure
 */
export interface ExtractedModelStructure {
  name: string;
  layers: ExtractedLayer[];
  connections: ExtractedConnection[];
  metadata?: Record<string, unknown>;
}
