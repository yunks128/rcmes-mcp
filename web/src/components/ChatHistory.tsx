import { useState, useCallback, useEffect, useRef } from 'react';
import type { ChatMessage } from '../types/events';

export interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: string;
  updatedAt: string;
}

const STORAGE_KEY = 'rcmes-chat-sessions';

function generateId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = (Math.random() * 16) | 0;
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
  });
}

export function loadSessions(): ChatSession[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    return JSON.parse(raw);
  } catch {
    return [];
  }
}

function persistSessions(sessions: ChatSession[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
}

function stripBase64(messages: ChatMessage[]): ChatMessage[] {
  // Strip large base64 data from messages but keep lightweight image URLs
  return messages.map(m => ({
    ...m,
    // Keep URL-based images, strip inline base64 data URIs
    images: m.images.filter(img => !img.startsWith('data:')),
    tools: m.tools.map(t => ({ ...t, image_base64: undefined })),
  }));
}

function deriveTitle(messages: ChatMessage[]): string {
  const firstUser = messages.find(m => m.role === 'user');
  if (!firstUser) return 'New Chat';
  const text = firstUser.content.slice(0, 50);
  return text.length < firstUser.content.length ? text + '...' : text;
}

/**
 * Auto-save a session to localStorage. Creates a new session if sessionId is null.
 * Returns the session ID (new or existing).
 */
export function autoSaveSession(
  messages: ChatMessage[],
  sessionId: string | null,
): string | null {
  if (messages.length === 0) return sessionId;

  const now = new Date().toISOString();
  const sessions = loadSessions();
  const stripped = stripBase64(messages);
  const title = deriveTitle(messages);

  if (sessionId) {
    const idx = sessions.findIndex(s => s.id === sessionId);
    if (idx >= 0) {
      sessions[idx] = { ...sessions[idx], title, messages: stripped, updatedAt: now };
    } else {
      // Session was deleted externally; create new
      sessions.unshift({ id: sessionId, title, messages: stripped, createdAt: now, updatedAt: now });
    }
    persistSessions(sessions);
    return sessionId;
  }

  // Create new session
  const newId = generateId();
  sessions.unshift({ id: newId, title, messages: stripped, createdAt: now, updatedAt: now });
  persistSessions(sessions);
  return newId;
}

function formatDate(dateStr: string): string {
  const d = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h ago`;
  const diffDays = Math.floor(diffHours / 24);
  if (diffDays < 7) return `${diffDays}d ago`;
  return d.toLocaleDateString();
}

interface ChatHistoryProps {
  open: boolean;
  onClose: () => void;
  currentSessionId: string | null;
  onLoadSession: (session: ChatSession) => void;
  onNewChat: () => void;
}

export default function ChatHistory({
  open,
  onClose,
  currentSessionId,
  onLoadSession,
  onNewChat,
}: ChatHistoryProps) {
  const [sessions, setSessions] = useState<ChatSession[]>(loadSessions);
  const closeBtnRef = useRef<HTMLButtonElement>(null);

  // Refresh sessions list when sidebar opens
  useEffect(() => {
    if (open) {
      setSessions(loadSessions());
    }
  }, [open]);

  // Escape to close
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    closeBtnRef.current?.focus();
    return () => document.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  const handleDelete = useCallback((id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const updated = loadSessions().filter(s => s.id !== id);
    persistSessions(updated);
    setSessions(updated);
  }, []);

  const handleLoad = useCallback((session: ChatSession) => {
    onLoadSession(session);
    onClose();
  }, [onLoadSession, onClose]);

  const handleNewChat = useCallback(() => {
    onNewChat();
    onClose();
  }, [onNewChat, onClose]);

  return (
    <div
      className={`chat-history-sidebar ${open ? 'chat-history-sidebar--open' : ''}`}
      role="dialog"
      aria-modal="true"
      aria-labelledby="chat-history-title"
      aria-hidden={!open}
    >
      <div className="chat-history-header">
        <h2 id="chat-history-title">Chat History</h2>
        <button
          ref={closeBtnRef}
          className="chat-history-close"
          onClick={onClose}
          aria-label="Close chat history"
        >
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" width="18" height="18" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <button className="chat-history-new" onClick={handleNewChat}>
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" width="18" height="18" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
        </svg>
        New Chat
      </button>

      <div className="chat-history-list">
        {sessions.length === 0 ? (
          <div className="chat-history-empty">No saved chats yet</div>
        ) : (
          sessions.map(session => (
            <div
              key={session.id}
              className={`chat-history-item ${session.id === currentSessionId ? 'chat-history-item--active' : ''}`}
              onClick={() => handleLoad(session)}
            >
              <div className="chat-history-item-title">{session.title}</div>
              <div className="chat-history-item-meta">
                {session.messages.length} messages &middot; {formatDate(session.updatedAt)}
              </div>
              <button
                className="chat-history-item-delete"
                onClick={(e) => handleDelete(session.id, e)}
                title="Delete"
                aria-label={`Delete chat: ${session.title}`}
              >
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" width="14" height="14" aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0" />
                </svg>
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
