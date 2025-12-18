import Ajv from 'ajv';
import schema from './nn3d.schema.json';
import type { NN3DModel } from './types';

// Create AJV instance with formats
const ajv = new Ajv({ allErrors: true, strict: false });

// Compile the schema
const validate = ajv.compile<NN3DModel>(schema);

/**
 * Validation result
 */
export interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
}

/**
 * Validation error details
 */
export interface ValidationError {
  path: string;
  message: string;
  keyword: string;
  params: Record<string, unknown>;
}

/**
 * Validate an NN3D model against the schema
 */
export function validateNN3DModel(model: unknown): ValidationResult {
  const valid = validate(model);
  
  if (valid) {
    return { valid: true, errors: [] };
  }
  
  const errors: ValidationError[] = (validate.errors || []).map(err => ({
    path: err.instancePath || '/',
    message: err.message || 'Unknown validation error',
    keyword: err.keyword,
    params: err.params as Record<string, unknown>,
  }));
  
  return { valid: false, errors };
}

/**
 * Parse and validate a JSON string as NN3D model
 */
export function parseNN3DModel(jsonString: string): { model: NN3DModel | null; validation: ValidationResult } {
  try {
    const parsed = JSON.parse(jsonString);
    const validation = validateNN3DModel(parsed);
    
    return {
      model: validation.valid ? parsed as NN3DModel : null,
      validation,
    };
  } catch (e) {
    return {
      model: null,
      validation: {
        valid: false,
        errors: [{
          path: '/',
          message: `JSON parse error: ${e instanceof Error ? e.message : 'Unknown error'}`,
          keyword: 'parse',
          params: {},
        }],
      },
    };
  }
}

/**
 * Validate model structure beyond schema (semantic validation)
 */
export function validateModelSemantics(model: NN3DModel): ValidationResult {
  const errors: ValidationError[] = [];
  
  // Check for duplicate node IDs
  const nodeIds = new Set<string>();
  for (const node of model.graph.nodes) {
    if (nodeIds.has(node.id)) {
      errors.push({
        path: `/graph/nodes/${node.id}`,
        message: `Duplicate node ID: ${node.id}`,
        keyword: 'uniqueId',
        params: { id: node.id },
      });
    }
    nodeIds.add(node.id);
  }
  
  // Validate edge references
  for (const edge of model.graph.edges) {
    if (!nodeIds.has(edge.source)) {
      errors.push({
        path: `/graph/edges/${edge.id || 'unknown'}`,
        message: `Edge source node not found: ${edge.source}`,
        keyword: 'nodeRef',
        params: { source: edge.source },
      });
    }
    if (!nodeIds.has(edge.target)) {
      errors.push({
        path: `/graph/edges/${edge.id || 'unknown'}`,
        message: `Edge target node not found: ${edge.target}`,
        keyword: 'nodeRef',
        params: { target: edge.target },
      });
    }
  }
  
  // Validate subgraph node references
  if (model.graph.subgraphs) {
    for (const subgraph of model.graph.subgraphs) {
      for (const nodeId of subgraph.nodes) {
        if (!nodeIds.has(nodeId)) {
          errors.push({
            path: `/graph/subgraphs/${subgraph.id}`,
            message: `Subgraph references non-existent node: ${nodeId}`,
            keyword: 'nodeRef',
            params: { nodeId },
          });
        }
      }
    }
  }
  
  // Check for cycles (optional - may be valid in some networks)
  const hasCycle = detectCycles(model.graph.nodes, model.graph.edges);
  if (hasCycle) {
    errors.push({
      path: '/graph',
      message: 'Graph contains cycles (may be intentional for recurrent networks)',
      keyword: 'cycle',
      params: {},
    });
  }
  
  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Detect cycles in the graph using DFS
 */
function detectCycles(nodes: { id: string }[], edges: { source: string; target: string }[]): boolean {
  const adjacency = new Map<string, string[]>();
  
  for (const node of nodes) {
    adjacency.set(node.id, []);
  }
  
  for (const edge of edges) {
    adjacency.get(edge.source)?.push(edge.target);
  }
  
  const visited = new Set<string>();
  const recursionStack = new Set<string>();
  
  function dfs(nodeId: string): boolean {
    visited.add(nodeId);
    recursionStack.add(nodeId);
    
    const neighbors = adjacency.get(nodeId) || [];
    for (const neighbor of neighbors) {
      if (!visited.has(neighbor)) {
        if (dfs(neighbor)) return true;
      } else if (recursionStack.has(neighbor)) {
        return true;
      }
    }
    
    recursionStack.delete(nodeId);
    return false;
  }
  
  for (const node of nodes) {
    if (!visited.has(node.id)) {
      if (dfs(node.id)) return true;
    }
  }
  
  return false;
}

export { schema as nn3dSchema };
