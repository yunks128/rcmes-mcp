/**
 * SSE Event types for chat streaming
 */

export interface MessageStartEvent {
  request_id: string;
}

export interface TextDeltaEvent {
  content: string;
}

export interface ToolStartEvent {
  tool_call_id: string;
  tool_name: string;
  args: Record<string, unknown>;
  inputs: string[];
}

export interface ToolCompleteEvent {
  tool_call_id: string;
  tool_name: string;
  result_summary: string;
  outputs: string[];
  duration_ms: number;
  status: string;
}

export interface ToolErrorEvent {
  tool_call_id: string;
  tool_name: string;
  error: string;
  duration_ms: number;
}

export interface ToolProgressEvent {
  tool_call_id: string;
  progress: { completed: number; total: number; detail: string };
}

export interface ImageEvent {
  tool_call_id: string;
  image_base64: string;
}

export interface MessageEndEvent {
  request_id: string;
}

export type SSEEventType =
  | 'message_start'
  | 'text_delta'
  | 'tool_start'
  | 'tool_complete'
  | 'tool_progress'
  | 'tool_error'
  | 'image'
  | 'heartbeat'
  | 'message_end';

export type ToolStatus = 'pending' | 'running' | 'done' | 'error';

export interface ToolExecution {
  tool_call_id: string;
  tool_name: string;
  args: Record<string, unknown>;
  inputs: string[];
  outputs: string[];
  status: ToolStatus;
  result_summary?: string;
  error?: string;
  duration_ms?: number;
  image_base64?: string;
  progress?: { completed: number; total: number; detail: string };
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  images: string[];
  tools: ToolExecution[];
  timestamp: Date;
  streaming?: boolean;
}
