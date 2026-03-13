import { useEffect, useMemo } from 'react';
import {
  ReactFlow,
  Background,
  useNodesState,
  useEdgesState,
  MarkerType,
  type Node,
  type Edge,
  type DefaultEdgeOptions,
} from '@xyflow/react';
import dagre from 'dagre';
import ToolNode from './ToolNode';
import '@xyflow/react/dist/style.css';

interface ToolDAGProps {
  nodes: Node[];
  edges: Edge[];
  visible: boolean;
  onToggle: () => void;
}

const nodeTypes = { toolNode: ToolNode };

const defaultEdgeOptions: DefaultEdgeOptions = {
  style: { strokeWidth: 2 },
  markerEnd: { type: MarkerType.ArrowClosed },
};

function getLayoutedElements(nodes: Node[], edges: Edge[]) {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: 'LR', nodesep: 50, ranksep: 80 });

  nodes.forEach(node => {
    g.setNode(node.id, { width: 220, height: 90 });
  });

  edges.forEach(edge => {
    g.setEdge(edge.source, edge.target);
  });

  dagre.layout(g);

  const layoutedNodes = nodes.map(node => {
    const nodeWithPosition = g.node(node.id);
    return {
      ...node,
      position: {
        x: nodeWithPosition.x - 80,
        y: nodeWithPosition.y - 35,
      },
    };
  });

  return { nodes: layoutedNodes, edges };
}

export default function ToolDAG({ nodes: inputNodes, edges: inputEdges, visible, onToggle }: ToolDAGProps) {
  const { nodes: layoutedNodes, edges: layoutedEdges } = useMemo(
    () => getLayoutedElements(inputNodes, inputEdges),
    [inputNodes, inputEdges]
  );

  const [nodes, setNodes, onNodesChange] = useNodesState(layoutedNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(layoutedEdges);

  useEffect(() => {
    const { nodes: ln, edges: le } = getLayoutedElements(inputNodes, inputEdges);
    setNodes(ln);
    setEdges(le);
  }, [inputNodes, inputEdges, setNodes, setEdges]);

  if (inputNodes.length === 0) return null;

  return (
    <div className="dag-container">
      <div className="dag-header" onClick={onToggle} style={{ cursor: 'pointer' }}>
        <span className="dag-toggle">{visible ? '\u25BC' : '\u25B6'}</span>
        <span className="dag-title">Tool Execution Pipeline</span>
        <span className="dag-count">{inputNodes.length} tool{inputNodes.length !== 1 ? 's' : ''}</span>
      </div>
      {visible && (
        <div className="dag-flow-wrapper">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            nodeTypes={nodeTypes}
            defaultEdgeOptions={defaultEdgeOptions}
            fitView
            fitViewOptions={{ padding: 0.3, minZoom: 0.4, maxZoom: 1.2 }}
            panOnDrag
            zoomOnScroll
            zoomOnDoubleClick
            preventScrolling={false}
            nodesDraggable={false}
            nodesConnectable={false}
            minZoom={0.3}
            maxZoom={2}
            proOptions={{ hideAttribution: true }}
          >
            <Background gap={16} size={1} color="#e0e0e0" />
          </ReactFlow>
        </div>
      )}
    </div>
  );
}
