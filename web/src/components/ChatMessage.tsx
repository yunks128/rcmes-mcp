import React, { useMemo, useState, useCallback, useRef } from 'react';
import type { ChatMessage as ChatMessageType, ToolStatus } from '../types/events';
import { formatMarkdown } from '../utils/markdown';

interface ChatMessageProps {
  message: ChatMessageType;
  onImageClick?: (src: string) => void;
  onOptionSelect?: (option: string) => void;
  isLatest?: boolean;
}

const STATUS_LABELS: Record<ToolStatus, string> = {
  pending: 'Pending',
  running: 'Running...',
  done: 'Done',
  error: 'Error',
};

const STATUS_COLORS: Record<ToolStatus, string> = {
  pending: '#9e9e9e',
  running: '#1976d2',
  done: '#4caf50',
  error: '#f44336',
};

function toolDisplayName(name: string): string {
  return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

interface ParsedOption {
  number: string;
  label: string;
  description: string;
}

interface OptionGroup {
  heading: string;
  options: ParsedOption[];
}

/**
 * Check if a bold line is a heading/question (not a selectable option).
 * Headings contain question marks, are long sentences, or start with "Which/What/How".
 */
function isBoldHeading(text: string): boolean {
  if (text.includes('?')) return true;
  if (/^(Which|What|How|Where|When|Choose|Select|Pick)\b/i.test(text)) return true;
  const words = text.trim().split(/\s+/);
  if (words.length >= 5) return true;
  return false;
}

/**
 * Strip markdown inline formatting (bold, backticks) from a string to get plain label.
 */
function stripInlineMarkdown(text: string): string {
  return text.replace(/\*\*(.+?)\*\*/g, '$1').replace(/`(.+?)`/g, '$1').trim();
}

/**
 * Extract option groups from the message content.
 * Each group is a bold heading (**...**) followed by options.
 * Options can be: numbered lines, bold/backtick labels (with optional description on next line),
 * or plain text lines under a heading.
 *
 * Heading lines are NOT consumed — they stay in the displayed text as section titles.
 * Only option lines (and their description lines) are consumed/removed.
 */
function extractOptionGroups(content: string): OptionGroup[] {
  const groups: OptionGroup[] = [];
  const lines = content.split('\n');
  let currentHeading = '';
  let currentOptions: ParsedOption[] = [];
  // Track which line indices are consumed as options for removal
  const consumedLines = new Set<number>();

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Check for bold line: **...**  possibly with trailing parenthetical
    const boldMatch = line.match(/^\s*\*\*(.+?)\*\*\s*(?:\((.+?)\))?\s*$/);
    if (boldMatch) {
      const boldText = boldMatch[1];

      if (isBoldHeading(boldText)) {
        // Save previous group
        if (currentOptions.length > 0) {
          groups.push({ heading: currentHeading, options: currentOptions });
          currentOptions = [];
        }
        currentHeading = boldText;
        // Do NOT add to consumedLines — headings stay in the rendered text
        continue;
      }

      // It's a bold option label (e.g. **tasmax** or **ssp126** (low emissions))
      if (currentHeading) {
        let description = boldMatch[2] || '';
        consumedLines.add(i);

        // Check if next non-empty line is a plain description
        if (!description) {
          const nextIdx = i + 1;
          if (nextIdx < lines.length) {
            const nextLine = lines[nextIdx].trim();
            if (nextLine && !nextLine.startsWith('**') && !nextLine.startsWith('`')
                && !nextLine.match(/^\d+[.)]\s/) && !nextLine.startsWith('#')
                && !isBoldHeading(stripInlineMarkdown(nextLine))) {
              description = nextLine;
              consumedLines.add(nextIdx);
              i++;
            }
          }
        }

        currentOptions.push({
          number: String(currentOptions.length + 1),
          label: boldText,
          description,
        });
        continue;
      }
    }

    // Check for backtick label line: `tasmax`  possibly with trailing parenthetical
    const backtickMatch = line.match(/^\s*`(.+?)`\s*(?:\((.+?)\))?\s*$/);
    if (backtickMatch && currentHeading) {
      const label = backtickMatch[1];
      let description = backtickMatch[2] || '';
      consumedLines.add(i);

      // Check if next line is a description
      if (!description) {
        const nextIdx = i + 1;
        if (nextIdx < lines.length) {
          const nextLine = lines[nextIdx].trim();
          if (nextLine && !nextLine.startsWith('**') && !nextLine.startsWith('`')
              && !nextLine.match(/^\d+[.)]\s/) && !nextLine.startsWith('#')) {
            description = nextLine;
            consumedLines.add(nextIdx);
            i++;
          }
        }
      }

      currentOptions.push({
        number: String(currentOptions.length + 1),
        label,
        description,
      });
      continue;
    }

    // Check for numbered option (e.g. "1. California (32-42N, 124-114W)")
    const optMatch = line.match(/^\s*(\d+)[.)]\s+(.+)$/);
    if (optMatch) {
      const fullText = optMatch[2].trim();
      const dashIdx = fullText.indexOf(' - ');
      // Get the label part (before dash) to check if it's a question
      const labelPart = stripInlineMarkdown(dashIdx >= 0 ? fullText.substring(0, dashIdx) : fullText);

      // Skip lines where the label itself is a question (e.g. "Which variable are you interested in?")
      // but allow action items like "Show a spatial map of the data" even if they're long
      const isQuestion = labelPart.endsWith('?') ||
        (/^(Which|What|How|Where|When)\b/i.test(labelPart) && labelPart.split(/\s+/).length >= 5);
      if (isQuestion) {
        // Treat as a heading for subsequent options
        if (currentOptions.length > 0) {
          groups.push({ heading: currentHeading, options: currentOptions });
          currentOptions = [];
        }
        currentHeading = labelPart.replace(/\?$/, '').trim();
        continue;
      }
      consumedLines.add(i);
      if (dashIdx >= 0) {
        currentOptions.push({
          number: optMatch[1],
          label: stripInlineMarkdown(fullText.substring(0, dashIdx)),
          description: fullText.substring(dashIdx + 3).trim(),
        });
      } else {
        currentOptions.push({
          number: optMatch[1],
          label: stripInlineMarkdown(fullText),
          description: '',
        });
      }
      continue;
    }

    // Plain non-empty line under a heading could be an option (e.g. "ACCESS-CM2")
    const trimmed = line.trim();
    if (currentHeading && trimmed && currentOptions.length > 0
        && !trimmed.startsWith('#') && !trimmed.startsWith('(')
        && !trimmed.startsWith('e.g') && !trimmed.startsWith('E.g')) {
      const words = trimmed.split(/\s+/);
      // Short labels without sentence-ending punctuation
      if (words.length <= 5 && !trimmed.endsWith('.') && !trimmed.endsWith('?')
          && !trimmed.endsWith(':') && !trimmed.endsWith('!')) {
        consumedLines.add(i);
        currentOptions.push({
          number: String(currentOptions.length + 1),
          label: stripInlineMarkdown(trimmed),
          description: '',
        });
        continue;
      }
    }
  }

  // Save last group
  if (currentOptions.length > 0) {
    groups.push({ heading: currentHeading, options: currentOptions });
  }

  // Attach consumed line indices to groups for removal
  (groups as any)._consumedLines = consumedLines;

  return groups;
}

interface ParseResult {
  groups: OptionGroup[];
  consumedLines: Set<number>;
}

function extractOptionGroupsWithConsumed(content: string): ParseResult {
  const result = extractOptionGroups(content);
  const consumedLines: Set<number> = (result as any)._consumedLines || new Set();
  return { groups: result, consumedLines };
}

/**
 * Remove lines that were parsed as options/headings from content
 * to avoid showing them twice (once as text, once as buttons).
 */
function removeOptionLines(content: string, consumed: Set<number>): string {
  if (consumed.size === 0) return content;

  return content
    .split('\n')
    .filter((_, idx) => !consumed.has(idx))
    .join('\n');
}

export default function ChatMessage({ message, onImageClick, onOptionSelect, isLatest }: ChatMessageProps) {
  const { role, content, images, tools, streaming } = message;

  // Only show option buttons on the latest assistant message that isn't streaming
  const parseResult = useMemo(() => {
    if (role !== 'assistant' || streaming || !isLatest) return { groups: [], consumedLines: new Set<number>() };
    return extractOptionGroupsWithConsumed(content);
  }, [role, content, streaming, isLatest]);

  const optionGroups = parseResult.groups;
  const totalOptions = optionGroups.reduce((sum, g) => sum + g.options.length, 0);
  const isMultiGroup = optionGroups.length > 1;
  const showOptionButtons = totalOptions >= 2 && onOptionSelect;

  // If we're showing buttons, render the text without the option lines
  const displayContent = showOptionButtons ? removeOptionLines(content, parseResult.consumedLines) : content;

  return (
    <div className={`chat-msg chat-msg--${role}`}>
      {role === 'assistant' && (
        <div className="chat-msg-avatar">AI</div>
      )}
      <div className="chat-msg-body">
        {/* Tool execution pipeline (resizable) */}
        {tools.length > 0 && (
          <ResizableToolPanel>
            {tools.map(tool => (
              tool.tool_name === 'execute_python_code' ? (
                <div key={tool.tool_call_id} className={`code-exec-badge code-exec-badge--${tool.status}`}>
                  <div className="code-exec-header">
                    <span className="code-exec-label">Python</span>
                    <span className="tool-badge-status" style={{ color: STATUS_COLORS[tool.status] }}>
                      {STATUS_LABELS[tool.status]}
                    </span>
                    {tool.duration_ms != null && tool.status !== 'running' && (
                      <span className="tool-badge-time" style={{ marginLeft: 'auto' }}>
                        {tool.duration_ms < 1000 ? `${tool.duration_ms}ms` : `${(tool.duration_ms / 1000).toFixed(1)}s`}
                      </span>
                    )}
                  </div>
                  <pre className="code-exec-source"><code>{String(tool.args.code || '')}</code></pre>
                  {tool.result_summary && tool.status === 'done' && (
                    <pre className="code-exec-output">{tool.result_summary}</pre>
                  )}
                  {tool.error && (
                    <pre className="code-exec-error">{tool.error}</pre>
                  )}
                  {tool.status === 'running' && (
                    <div className="code-exec-running">Running...</div>
                  )}
                </div>
              ) : (
                <div
                  key={tool.tool_call_id}
                  className={`tool-badge tool-badge--${tool.status}`}
                  style={{ borderLeftColor: STATUS_COLORS[tool.status] }}
                >
                  <div className="tool-badge-header">
                    <span className="tool-badge-name">{toolDisplayName(tool.tool_name)}</span>
                    <span className="tool-badge-status" style={{ color: STATUS_COLORS[tool.status] }}>
                      {STATUS_LABELS[tool.status]}
                    </span>
                  </div>
                  {tool.result_summary && tool.status === 'done' && (
                    <div className="tool-badge-summary" dangerouslySetInnerHTML={{ __html: formatMarkdown(tool.result_summary) }} />
                  )}
                  {tool.progress && tool.status === 'running' && (
                    <div className="tool-badge-progress">
                      <div className="tool-badge-progress-detail">
                        {tool.progress.detail || `Step ${tool.progress.completed}/${tool.progress.total}`}
                      </div>
                      <div className="tool-badge-progress-bar-track">
                        {tool.progress.total > 0 ? (
                          <div
                            className="tool-badge-progress-bar-fill"
                            style={{
                              width: `${(tool.progress.completed / tool.progress.total) * 100}%`,
                              background: STATUS_COLORS[tool.status],
                            }}
                          />
                        ) : (
                          <div className="tool-badge-progress-bar-indeterminate" />
                        )}
                      </div>
                      {tool.progress.total > 0 && (
                        <div className="tool-badge-progress-step">
                          {tool.progress.completed}/{tool.progress.total}
                        </div>
                      )}
                    </div>
                  )}
                  {tool.error && (
                    <div className="tool-badge-error">{tool.error}</div>
                  )}
                  {tool.duration_ms != null && tool.status !== 'running' && (
                    <div className="tool-badge-time">
                      {tool.duration_ms < 1000 ? `${tool.duration_ms}ms` : `${(tool.duration_ms / 1000).toFixed(1)}s`}
                    </div>
                  )}
                </div>
              )
            ))}
          </ResizableToolPanel>
        )}

        {/* Text content */}
        {displayContent && (
          <div
            className="chat-msg-content"
            dangerouslySetInnerHTML={{ __html: formatMarkdown(displayContent) }}
          />
        )}

        {/* Option buttons: single-select (instant) or multi-select (with submit) */}
        {showOptionButtons && (
          (isMultiGroup || /select multiple|choose multiple|pick multiple|can select multiple/i.test(content))
            ? <MultiSelectOptions groups={optionGroups} onSubmit={onOptionSelect} />
            : <div className="option-buttons">
                {optionGroups[0].options.map(opt => (
                  <button
                    key={opt.number}
                    className="option-btn"
                    onClick={() => onOptionSelect(opt.label)}
                  >
                    <span className="option-btn-label">{opt.label}</span>
                    {opt.description && (
                      <span className="option-btn-desc">{opt.description}</span>
                    )}
                  </button>
                ))}
              </div>
        )}

        {/* Streaming indicator */}
        {streaming && !content && tools.every(t => t.status === 'running' || t.status === 'pending') && (
          <div className="chat-typing">
            <span></span><span></span><span></span>
          </div>
        )}

        {/* Inline images */}
        {images.map((img, i) => {
          // Support both server URLs (/api/images/...) and legacy base64 data URIs
          const src = img.startsWith('data:') || img.startsWith('/') || img.startsWith('http')
            ? img
            : `data:image/png;base64,${img}`;
          return (
            <img
              key={i}
              className="chat-msg-image"
              src={src}
              alt="Visualization"
              onClick={() => onImageClick?.(src)}
            />
          );
        })}
      </div>
    </div>
  );
}

function ResizableToolPanel({ children }: { children: React.ReactNode }) {
  const panelRef = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState<number | null>(null);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const dragging = useRef(false);
  const startY = useRef(0);
  const startH = useRef(0);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    dragging.current = true;
    startY.current = e.clientY;
    startH.current = panelRef.current?.offsetHeight || 200;

    const onMouseMove = (ev: MouseEvent) => {
      if (!dragging.current) return;
      const delta = ev.clientY - startY.current;
      const newH = Math.max(48, startH.current + delta);
      setHeight(newH);
    };
    const onMouseUp = () => {
      dragging.current = false;
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }, []);

  const toggleCollapse = useCallback(() => {
    setIsCollapsed(prev => !prev);
  }, []);

  return (
    <div
      ref={panelRef}
      className={`chat-msg-tools ${isCollapsed ? 'chat-msg-tools--collapsed' : ''}`}
      style={!isCollapsed && height ? { height, maxHeight: height } : undefined}
    >
      <div className="tools-panel-controls">
        <button className="tools-panel-toggle" onClick={toggleCollapse} title={isCollapsed ? 'Expand' : 'Collapse'}>
          {isCollapsed ? '\u25B6' : '\u25BC'} Tools ({React.Children.count(children)})
        </button>
      </div>
      {!isCollapsed && (
        <>
          <div className="tools-panel-content">
            {children}
          </div>
          <div className="tools-panel-resize-handle" onMouseDown={onMouseDown} title="Drag to resize">
            <span className="tools-panel-resize-grip" />
          </div>
        </>
      )}
    </div>
  );
}


function MultiSelectOptions({ groups, onSubmit }: { groups: OptionGroup[]; onSubmit: (option: string) => void }) {
  const [selected, setSelected] = useState<Record<string, string>>({});

  const toggle = useCallback((groupHeading: string, label: string) => {
    setSelected(prev => {
      const next = { ...prev };
      if (next[groupHeading] === label) {
        delete next[groupHeading];
      } else {
        next[groupHeading] = label;
      }
      return next;
    });
  }, []);

  const handleSubmit = useCallback(() => {
    const selections = groups
      .map(g => selected[g.heading])
      .filter(Boolean);
    if (selections.length === 0) return;
    onSubmit(selections.join(', '));
  }, [groups, selected, onSubmit]);

  const selectionCount = Object.keys(selected).length;

  return (
    <div className="multi-select-options">
      {groups.map(group => (
        <div key={group.heading} className="option-group">
          {group.heading && (
            <div className="option-group-heading">{group.heading}</div>
          )}
          <div className="option-buttons">
            {group.options.map(opt => {
              const isSelected = selected[group.heading] === opt.label;
              return (
                <button
                  key={opt.number}
                  className={`option-btn ${isSelected ? 'option-btn--selected' : ''}`}
                  onClick={() => toggle(group.heading, opt.label)}
                >
                  <span className="option-btn-label">{opt.label}</span>
                  {opt.description && (
                    <span className="option-btn-desc">{opt.description}</span>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      ))}
      <button
        className="option-submit-btn"
        onClick={handleSubmit}
        disabled={selectionCount === 0}
      >
        Submit{selectionCount > 0 ? ` (${selectionCount} selected)` : ''}
      </button>
    </div>
  );
}
