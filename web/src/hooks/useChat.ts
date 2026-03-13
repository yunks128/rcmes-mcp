import { useState, useCallback, useRef } from 'react';
import { fetchEventSource } from '@microsoft/fetch-event-source';
import type {
  ChatMessage,
  ToolExecution,
  TextDeltaEvent,
  ToolStartEvent,
  ToolCompleteEvent,
  ToolProgressEvent,
  ToolErrorEvent,
  ImageEvent,
} from '../types/events';

const API_BASE = '/api';

function generateId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return generateId();
  }
  // Fallback for non-secure contexts (HTTP)
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = (Math.random() * 16) | 0;
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
  });
}

interface UseChatOptions {
  onToolStart?: (tool: ToolExecution) => void;
  onToolComplete?: (tool: ToolExecution) => void;
  onToolProgress?: (tool: ToolExecution) => void;
  onMessageEnd?: () => void;
}

export function useChat(options: UseChatOptions = {}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const toolMapRef = useRef<Map<string, ToolExecution>>(new Map());
  // Use refs to avoid stale closures in sendMessage
  const streamingRef = useRef(false);
  const messagesRef = useRef<ChatMessage[]>([]);
  const optionsRef = useRef(options);
  optionsRef.current = options;

  // Keep refs in sync
  const updateMessages = useCallback((updater: (prev: ChatMessage[]) => ChatMessage[]) => {
    setMessages(prev => {
      const next = updater(prev);
      messagesRef.current = next;
      return next;
    });
  }, []);

  const sendMessage = useCallback(async (content: string, datasetId?: string) => {
    if (!content.trim() || streamingRef.current) return;

    const userMsg: ChatMessage = {
      id: generateId(),
      role: 'user',
      content,
      images: [],
      tools: [],
      timestamp: new Date(),
    };

    const assistantMsg: ChatMessage = {
      id: generateId(),
      role: 'assistant',
      content: '',
      images: [],
      tools: [],
      timestamp: new Date(),
      streaming: true,
    };

    // Capture history BEFORE adding new messages
    const history = [...messagesRef.current, userMsg].map(m => ({
      role: m.role,
      content: m.content,
    }));

    updateMessages(prev => [...prev, userMsg, assistantMsg]);
    streamingRef.current = true;
    setStreaming(true);
    toolMapRef.current.clear();

    const abortController = new AbortController();
    abortRef.current = abortController;

    const markDone = (extraContent?: string) => {
      updateMessages(prev => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last?.role === 'assistant') {
          updated[updated.length - 1] = {
            ...last,
            content: extraContent ? last.content + extraContent : last.content,
            streaming: false,
          };
        }
        return updated;
      });
      streamingRef.current = false;
      setStreaming(false);
      abortRef.current = null;
    };

    try {
      await fetchEventSource(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: history,
          dataset_id: datasetId || null,
        }),
        signal: abortController.signal,

        async onopen(response) {
          if (!response.ok) {
            const text = await response.text().catch(() => response.statusText);
            throw new Error(`Server error ${response.status}: ${text}`);
          }
        },

        onmessage(ev) {
          if (!ev.event || ev.event === 'heartbeat') return;

          let data: any;
          try {
            data = JSON.parse(ev.data);
          } catch {
            return;
          }

          switch (ev.event) {
            case 'text_delta': {
              const { content: delta } = data as TextDeltaEvent;
              updateMessages(prev => {
                const updated = [...prev];
                const last = updated[updated.length - 1];
                if (last?.role === 'assistant') {
                  updated[updated.length - 1] = { ...last, content: last.content + delta };
                }
                return updated;
              });
              break;
            }

            case 'tool_start': {
              const evt = data as ToolStartEvent;
              const tool: ToolExecution = {
                tool_call_id: evt.tool_call_id,
                tool_name: evt.tool_name,
                args: evt.args,
                inputs: evt.inputs,
                outputs: [],
                status: 'running',
              };
              toolMapRef.current.set(evt.tool_call_id, tool);
              updateMessages(prev => {
                const updated = [...prev];
                const last = updated[updated.length - 1];
                if (last?.role === 'assistant') {
                  updated[updated.length - 1] = {
                    ...last,
                    tools: [...last.tools, tool],
                  };
                }
                return updated;
              });
              optionsRef.current.onToolStart?.(tool);
              break;
            }

            case 'tool_progress': {
              const evt = data as ToolProgressEvent;
              const tool = toolMapRef.current.get(evt.tool_call_id);
              if (tool) {
                tool.progress = evt.progress;
              }
              updateMessages(prev => {
                const updated = [...prev];
                const last = updated[updated.length - 1];
                if (last?.role === 'assistant') {
                  updated[updated.length - 1] = {
                    ...last,
                    tools: last.tools.map(t =>
                      t.tool_call_id === evt.tool_call_id
                        ? { ...t, progress: evt.progress }
                        : t
                    ),
                  };
                }
                return updated;
              });
              if (tool) optionsRef.current.onToolProgress?.(tool);
              break;
            }

            case 'tool_complete': {
              const evt = data as ToolCompleteEvent;
              const tool = toolMapRef.current.get(evt.tool_call_id);
              if (tool) {
                tool.status = 'done';
                tool.result_summary = evt.result_summary;
                tool.outputs = evt.outputs;
                tool.duration_ms = evt.duration_ms;
              }
              updateMessages(prev => {
                const updated = [...prev];
                const last = updated[updated.length - 1];
                if (last?.role === 'assistant') {
                  updated[updated.length - 1] = {
                    ...last,
                    tools: last.tools.map(t =>
                      t.tool_call_id === evt.tool_call_id
                        ? { ...t, status: 'done' as const, result_summary: evt.result_summary, outputs: evt.outputs, duration_ms: evt.duration_ms }
                        : t
                    ),
                  };
                }
                return updated;
              });
              if (tool) optionsRef.current.onToolComplete?.(tool);
              break;
            }

            case 'tool_error': {
              const evt = data as ToolErrorEvent;
              updateMessages(prev => {
                const updated = [...prev];
                const last = updated[updated.length - 1];
                if (last?.role === 'assistant') {
                  updated[updated.length - 1] = {
                    ...last,
                    tools: last.tools.map(t =>
                      t.tool_call_id === evt.tool_call_id
                        ? { ...t, status: 'error' as const, error: evt.error, duration_ms: evt.duration_ms }
                        : t
                    ),
                  };
                }
                return updated;
              });
              break;
            }

            case 'image': {
              const evt = data as ImageEvent;
              updateMessages(prev => {
                const updated = [...prev];
                const last = updated[updated.length - 1];
                if (last?.role === 'assistant') {
                  updated[updated.length - 1] = {
                    ...last,
                    images: [...last.images, evt.image_base64],
                    tools: last.tools.map(t =>
                      t.tool_call_id === evt.tool_call_id
                        ? { ...t, image_base64: evt.image_base64 }
                        : t
                    ),
                  };
                }
                return updated;
              });
              break;
            }

            case 'message_end': {
              markDone();
              optionsRef.current.onMessageEnd?.();
              break;
            }
          }
        },

        onerror(err) {
          if (abortController.signal.aborted) return;
          console.error('SSE error:', err);
          markDone('\n\n*Connection error. Please try again.*');
          throw err; // Stop retrying
        },

        onclose() {
          // Server closed the connection — mark done if still streaming
          if (streamingRef.current) {
            markDone();
          }
        },

        openWhenHidden: true,
      });
    } catch (err) {
      if (abortController.signal.aborted) return;
      console.error('Chat stream error:', err);
      // If still streaming, show the error to the user
      if (streamingRef.current) {
        const errMsg = err instanceof Error ? err.message : 'Failed to connect';
        markDone(`\n\n*Error: ${errMsg}*`);
      }
    }
  }, [updateMessages]); // Minimal dependencies — uses refs for everything else

  const cancelStream = useCallback(() => {
    abortRef.current?.abort();
    streamingRef.current = false;
    setStreaming(false);
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([]);
    messagesRef.current = [];
  }, []);

  const loadMessages = useCallback((msgs: ChatMessage[]) => {
    setMessages(msgs);
    messagesRef.current = msgs;
  }, []);

  return { messages, streaming, sendMessage, cancelStream, clearMessages, loadMessages };
}
