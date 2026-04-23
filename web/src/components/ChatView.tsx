import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useChat } from '../hooks/useChat';
import { useDAG } from '../hooks/useDAG';
import ChatMessageComponent from './ChatMessage';
import ChatInput from './ChatInput';
import ToolDAG from './ToolDAG';
import ImageViewer from './ImageViewer';
import SettingsDrawer from './SettingsDrawer';
import ChatHistory, { autoSaveSession } from './ChatHistory';
import type { ChatMessage, ToolExecution } from '../types/events';
import type { ChatSession } from './ChatHistory';

const WELCOME_MESSAGE: ChatMessage = {
  id: 'welcome',
  role: 'assistant',
  content:
    "Welcome! I'm the RCMES (Regional Climate Model Evaluation System) Climate Assistant. I can help you explore NASA's NEX-GDDP-CMIP6 (NASA Earth Exchange Global Daily Downscaled Projections, Coupled Model Intercomparison Project Phase 6) climate projections — load data, run analyses, and create visualizations.\n\n" +
    "**What would you like to explore?**\n" +
    "1. Temperature trends - How will temperatures change?\n" +
    "2. Precipitation patterns - Future rainfall analysis\n" +
    "3. Extreme events - Heatwaves, droughts, heavy rain\n" +
    "4. Scenario comparison - Compare low vs high emissions\n" +
    "5. Country/region study - Focus on a specific area\n" +
    "6. I have a specific question - Just ask!",
  images: [],
  tools: [],
  timestamp: new Date(),
};

export default function ChatView() {
  const [lightboxSrc, setLightboxSrc] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [showScrollBtn, setShowScrollBtn] = useState(false);
  const sessionIdRef = useRef<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesAreaRef = useRef<HTMLDivElement>(null);
  const isAtBottomRef = useRef(true);
  const prevMsgCountRef = useRef(0);
  const { dag, addTool, updateTool, resetDAG, toggleVisible } = useDAG();

  // Keep ref in sync
  sessionIdRef.current = currentSessionId;

  const chatOptions = useMemo(() => ({
    onToolStart: (tool: ToolExecution) => addTool(tool),
    onToolComplete: (tool: ToolExecution) => updateTool(tool),
    onToolProgress: (tool: ToolExecution) => updateTool(tool),
    onMessageEnd: () => {
      // Auto-save is triggered via the streaming effect below
    },
  }), [addTool, updateTool]);

  const { messages, streaming, sendMessage, cancelStream, clearMessages, loadMessages } = useChat(chatOptions);

  // Auto-save when streaming ends (message complete)
  const prevStreamingRef = useRef(false);
  useEffect(() => {
    // Detect streaming -> not streaming transition (message just finished)
    if (prevStreamingRef.current && !streaming && messages.length > 0) {
      const newId = autoSaveSession(messages, sessionIdRef.current);
      if (newId && newId !== sessionIdRef.current) {
        setCurrentSessionId(newId);
      }
    }
    prevStreamingRef.current = streaming;
  }, [streaming, messages]);

  // Prepend the welcome message if no real messages exist
  const displayMessages = useMemo(() => {
    if (messages.length === 0) return [WELCOME_MESSAGE];
    return messages;
  }, [messages]);

  // Track whether the user is near the bottom so auto-scroll doesn't fight them.
  useEffect(() => {
    const el = messagesAreaRef.current;
    if (!el) return;
    const onScroll = () => {
      const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
      const atBottom = distance < 120;
      isAtBottomRef.current = atBottom;
      setShowScrollBtn(!atBottom);
    };
    el.addEventListener('scroll', onScroll, { passive: true });
    return () => el.removeEventListener('scroll', onScroll);
  }, []);

  // Scroll on: user just sent (msg count increased and last is user) OR already at bottom.
  useEffect(() => {
    const lengthIncreased = displayMessages.length > prevMsgCountRef.current;
    prevMsgCountRef.current = displayMessages.length;
    const last = displayMessages[displayMessages.length - 1];
    const userJustSent = lengthIncreased && last?.role === 'user';
    if (userJustSent || isAtBottomRef.current) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [displayMessages]);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  const handleSend = useCallback((content: string) => {
    resetDAG();
    sendMessage(content);
  }, [sendMessage, resetDAG]);

  const handleOptionSelect = useCallback((option: string) => {
    if (streaming) return;
    resetDAG();
    sendMessage(option);
  }, [sendMessage, streaming, resetDAG]);

  const handleNewChat = useCallback(() => {
    // Auto-save current before clearing
    if (messages.length > 0) {
      const newId = autoSaveSession(messages, currentSessionId);
      if (newId && newId !== currentSessionId) {
        // Session was saved with a new ID; no need to keep it
      }
    }
    clearMessages();
    resetDAG();
    setCurrentSessionId(null);
  }, [clearMessages, resetDAG, messages, currentSessionId]);

  const handleLoadSession = useCallback((session: ChatSession) => {
    loadMessages(session.messages);
    setCurrentSessionId(session.id);
    resetDAG();
  }, [loadMessages, resetDAG]);

  return (
    <div className="chat-layout">
      {/* History sidebar */}
      <ChatHistory
        open={historyOpen}
        onClose={() => setHistoryOpen(false)}
        currentSessionId={currentSessionId}
        onLoadSession={handleLoadSession}
        onNewChat={handleNewChat}
      />

      {/* Overlay when sidebar is open on mobile */}
      {historyOpen && <div className="chat-history-overlay" onClick={() => setHistoryOpen(false)} />}

      <div className="chat-view">
        {/* Top bar */}
        <header className="chat-topbar">
          <button
            className="topbar-btn"
            onClick={() => setHistoryOpen(!historyOpen)}
            title="Chat History"
            aria-label="Toggle chat history"
            aria-expanded={historyOpen}
          >
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" width="20" height="20" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
            </svg>
          </button>
          <h1 className="topbar-title">RCMES (Regional Climate Model Evaluation System) Climate Assistant</h1>
          <div className="topbar-actions">
            <button
              className="topbar-btn"
              onClick={() => setSettingsOpen(true)}
              title="Settings & Data"
              aria-label="Open settings and data"
            >
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" width="20" height="20" aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.325.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 0 1 1.37.49l1.296 2.247a1.125 1.125 0 0 1-.26 1.431l-1.003.827c-.293.241-.438.613-.43.992a7.723 7.723 0 0 1 0 .255c-.008.378.137.75.43.991l1.004.827c.424.35.534.955.26 1.43l-1.298 2.247a1.125 1.125 0 0 1-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.47 6.47 0 0 1-.22.128c-.331.183-.581.495-.644.869l-.213 1.281c-.09.543-.56.94-1.11.94h-2.594c-.55 0-1.019-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 0 1-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 0 1-1.369-.49l-1.297-2.247a1.125 1.125 0 0 1 .26-1.431l1.004-.827c.292-.24.437-.613.43-.991a6.932 6.932 0 0 1 0-.255c.007-.38-.138-.751-.43-.992l-1.004-.827a1.125 1.125 0 0 1-.26-1.43l1.297-2.247a1.125 1.125 0 0 1 1.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.086.22-.128.332-.183.582-.495.644-.869l.214-1.28Z" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
              </svg>
            </button>
            <button
              className="topbar-btn"
              onClick={handleNewChat}
              title="New Chat"
              aria-label="Start a new chat"
            >
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" width="20" height="20" aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
              </svg>
            </button>
            <a
              href="https://github.com/yunks128/rcmes-mcp"
              target="_blank"
              rel="noopener noreferrer"
              className="topbar-btn"
              title="GitHub"
              aria-label="Open GitHub repository in new tab"
            >
              <svg viewBox="0 0 16 16" width="18" height="18" fill="currentColor" aria-hidden="true">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
              </svg>
            </a>
          </div>
        </header>

        {/* DAG section */}
        <ToolDAG
          nodes={dag.nodes}
          edges={dag.edges}
          visible={dag.visible}
          onToggle={toggleVisible}
        />

        {/* Messages area */}
        <div className="chat-messages-area" ref={messagesAreaRef}>
          {displayMessages.map((msg, idx) => (
            <ChatMessageComponent
              key={msg.id}
              message={msg}
              onImageClick={setLightboxSrc}
              onOptionSelect={handleOptionSelect}
              isLatest={idx === displayMessages.length - 1 && !streaming}
            />
          ))}
          <div ref={messagesEndRef} />
        </div>

        {showScrollBtn && (
          <button
            className="chat-scroll-btn"
            onClick={scrollToBottom}
            aria-label="Scroll to latest message"
            title="Scroll to latest"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" strokeWidth={2} stroke="currentColor" width="18" height="18" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 13.5 12 21m0 0-7.5-7.5M12 21V3" />
            </svg>
          </button>
        )}

        {/* Input */}
        <ChatInput
          onSend={handleSend}
          disabled={streaming}
          onCancel={cancelStream}
          streaming={streaming}
        />

        {/* Lightbox */}
        <ImageViewer src={lightboxSrc} onClose={() => setLightboxSrc(null)} />

        {/* Settings drawer */}
        <SettingsDrawer open={settingsOpen} onClose={() => setSettingsOpen(false)} />
      </div>
    </div>
  );
}
