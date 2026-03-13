import { memo } from 'react';
import { Handle, Position } from '@xyflow/react';
import type { ToolStatus } from '../types/events';

interface ToolNodeData {
  label: string;
  toolName: string;
  status: ToolStatus;
  summary: string;
  error: string;
  duration_ms?: number;
  progress?: { completed: number; total: number; detail: string };
  [key: string]: unknown;
}

const STATUS_CONFIG: Record<ToolStatus, { color: string; bg: string; icon: string }> = {
  pending: { color: '#9e9e9e', bg: '#f5f5f5', icon: '\u25CB' },   // circle
  running: { color: '#1976d2', bg: '#e3f2fd', icon: '\u25D4' },   // half circle
  done:    { color: '#4caf50', bg: '#e8f5e9', icon: '\u2713' },   // checkmark
  error:   { color: '#f44336', bg: '#ffebee', icon: '\u2717' },   // x
};

function ToolNodeComponent({ data }: { data: ToolNodeData }) {
  const { label, status, summary, error, duration_ms } = data;
  const config = STATUS_CONFIG[status];

  return (
    <div
      className={`tool-node tool-node--${status}`}
      style={{
        border: `2px solid ${config.color}`,
        background: config.bg,
        borderRadius: 10,
        padding: '12px 18px',
        minWidth: 200,
        maxWidth: 300,
        fontSize: '0.95rem',
        boxShadow: status === 'running' ? `0 0 12px ${config.color}40` : '0 2px 6px rgba(0,0,0,0.1)',
      }}
    >
      <Handle type="target" position={Position.Left} style={{ background: config.color, width: 10, height: 10 }} />
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ fontSize: '1.2rem' }}>{config.icon}</span>
        <div>
          <strong style={{ color: config.color, fontSize: '0.95rem' }}>{label}</strong>
          <div style={{ fontSize: '0.75rem', color: '#888', fontFamily: 'monospace' }}>{data.toolName}</div>
        </div>
      </div>
      {summary && status === 'done' && (
        <div style={{ fontSize: '0.82rem', color: '#555', marginTop: 6, lineHeight: 1.3 }}>{summary}</div>
      )}
      {error && (
        <div style={{ fontSize: '0.82rem', color: '#f44336', marginTop: 6, lineHeight: 1.3 }}>{error}</div>
      )}
      {duration_ms != null && status !== 'running' && (
        <div style={{ fontSize: '0.78rem', color: '#888', marginTop: 4 }}>
          {duration_ms < 1000 ? `${duration_ms}ms` : `${(duration_ms / 1000).toFixed(1)}s`}
        </div>
      )}
      {status === 'running' && data.progress ? (
        <div style={{ marginTop: 6 }}>
          <div style={{ fontSize: '0.82rem', color: config.color }}>
            Downloading {data.progress.completed}/{data.progress.total} files
          </div>
          <div style={{
            marginTop: 4,
            height: 6,
            borderRadius: 3,
            background: '#e0e0e0',
            overflow: 'hidden',
          }}>
            <div style={{
              height: '100%',
              width: `${(data.progress.completed / data.progress.total) * 100}%`,
              background: config.color,
              borderRadius: 3,
              transition: 'width 0.3s ease',
            }} />
          </div>
          {data.progress.detail && (
            <div style={{ fontSize: '0.72rem', color: '#999', marginTop: 2, fontFamily: 'monospace' }}>
              {data.progress.detail}
            </div>
          )}
        </div>
      ) : status === 'running' ? (
        <div style={{ fontSize: '0.85rem', color: config.color, marginTop: 6 }}>Running...</div>
      ) : null}
      <Handle type="source" position={Position.Right} style={{ background: config.color, width: 10, height: 10 }} />
    </div>
  );
}

export default memo(ToolNodeComponent);
