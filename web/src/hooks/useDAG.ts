import { useCallback, useState } from 'react';
import type { Node, Edge } from '@xyflow/react';
import type { ToolExecution, ToolStatus } from '../types/events';

interface DAGState {
  nodes: Node[];
  edges: Edge[];
  visible: boolean;
}

const STATUS_COLORS: Record<ToolStatus, string> = {
  pending: '#9e9e9e',
  running: '#1976d2',
  done: '#4caf50',
  error: '#f44336',
};

function toolDisplayName(name: string): string {
  return name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase());
}

export function useDAG() {
  const [dag, setDAG] = useState<DAGState>({ nodes: [], edges: [], visible: false });

  const addTool = useCallback((tool: ToolExecution) => {
    setDAG(prev => {
      const nodeId = tool.tool_call_id;
      const existingIdx = prev.nodes.findIndex(n => n.id === nodeId);

      if (existingIdx >= 0) {
        // Update existing node
        const updated = [...prev.nodes];
        updated[existingIdx] = {
          ...updated[existingIdx],
          data: {
            ...updated[existingIdx].data,
            status: tool.status,
            summary: tool.result_summary || '',
            error: tool.error || '',
            duration_ms: tool.duration_ms,
            progress: tool.progress,
          },
        };
        return { ...prev, nodes: updated };
      }

      // Add new node
      const newNode: Node = {
        id: nodeId,
        type: 'toolNode',
        position: { x: 0, y: 0 }, // Will be laid out by dagre
        data: {
          label: toolDisplayName(tool.tool_name),
          toolName: tool.tool_name,
          status: tool.status,
          summary: tool.result_summary || '',
          error: tool.error || '',
          duration_ms: tool.duration_ms,
          args: tool.args,
        },
      };

      // Create edges from input dataset IDs to this node
      const newEdges: Edge[] = [];
      let hasDataEdge = false;
      for (const inputId of tool.inputs) {
        // Find nodes that output this dataset_id
        for (const existingNode of prev.nodes) {
          const nodeOutputs = (existingNode.data as any).outputs || [];
          if (nodeOutputs.includes(inputId)) {
            hasDataEdge = true;
            newEdges.push({
              id: `${existingNode.id}-${nodeId}`,
              source: existingNode.id,
              target: nodeId,
              animated: tool.status === 'running',
              style: { stroke: STATUS_COLORS[tool.status], strokeWidth: 2 },
              markerEnd: { type: 'arrowclosed' as any, color: STATUS_COLORS[tool.status] },
            });
          }
        }
      }

      // If no data edges, chain sequentially from the last node
      if (!hasDataEdge && prev.nodes.length > 0) {
        const lastNode = prev.nodes[prev.nodes.length - 1];
        newEdges.push({
          id: `${lastNode.id}-${nodeId}`,
          source: lastNode.id,
          target: nodeId,
          animated: tool.status === 'running',
          style: { stroke: STATUS_COLORS[tool.status], strokeWidth: 2 },
          markerEnd: { type: 'arrowclosed' as any, color: STATUS_COLORS[tool.status] },
        });
      }

      return {
        nodes: [...prev.nodes, newNode],
        edges: [...prev.edges, ...newEdges],
        visible: true,
      };
    });
  }, []);

  const updateTool = useCallback((tool: ToolExecution) => {
    setDAG(prev => {
      const nodeId = tool.tool_call_id;
      const nodes = prev.nodes.map(n => {
        if (n.id !== nodeId) return n;
        return {
          ...n,
          data: {
            ...n.data,
            status: tool.status,
            summary: tool.result_summary || '',
            error: tool.error || '',
            duration_ms: tool.duration_ms,
            outputs: tool.outputs,
            progress: tool.progress,
          },
        };
      });

      // Update edge animations and add new edges from outputs
      let edges = prev.edges.map(e => {
        if (e.target === nodeId) {
          return {
            ...e,
            animated: tool.status === 'running',
            style: { stroke: STATUS_COLORS[tool.status], strokeWidth: 2 },
            markerEnd: { type: 'arrowclosed' as any, color: STATUS_COLORS[tool.status] },
          };
        }
        return e;
      });

      return { ...prev, nodes, edges };
    });
  }, []);

  const resetDAG = useCallback(() => {
    setDAG({ nodes: [], edges: [], visible: false });
  }, []);

  const toggleVisible = useCallback(() => {
    setDAG(prev => ({ ...prev, visible: !prev.visible }));
  }, []);

  return { dag, addTool, updateTool, resetDAG, toggleVisible };
}
