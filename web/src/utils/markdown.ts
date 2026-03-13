/**
 * Simple markdown to HTML formatter with table support
 */
export function formatMarkdown(text: string): string {
  const lines = text.split('\n');
  const result: string[] = [];
  let inTable = false;
  let tableRows: string[] = [];

  for (const line of lines) {
    if (line.trim().startsWith('|') && line.trim().endsWith('|')) {
      if (!inTable) {
        inTable = true;
        tableRows = [];
      }
      if (!line.match(/^\|[\s\-:|]+\|$/)) {
        const cells = line.split('|').filter(Boolean);
        if (tableRows.length === 0) {
          tableRows.push(`<tr>${cells.map(c => `<th>${c.trim()}</th>`).join('')}</tr>`);
        } else {
          tableRows.push(`<tr>${cells.map(c => `<td>${c.trim()}</td>`).join('')}</tr>`);
        }
      }
    } else {
      if (inTable) {
        result.push(`<table class="stats-table">${tableRows.join('')}</table>`);
        inTable = false;
        tableRows = [];
      }
      result.push(line);
    }
  }

  if (inTable && tableRows.length > 0) {
    result.push(`<table class="stats-table">${tableRows.join('')}</table>`);
  }

  return result.join('\n')
    .replace(/## (.*?)(\n|$)/g, '<h3>$1</h3>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/`(.*?)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br />');
}
