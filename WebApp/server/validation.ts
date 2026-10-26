import Papa from 'papaparse';
import type { ImportPreview, MessageInput, SessionInput, Source } from '../shared/types.ts';

export class HttpError extends Error {
  constructor(public status: number, message: string, public details?: string[]) { super(message); }
}
const MAX_BYTES = 5 * 1024 * 1024;
const sources = new Set<Source>(['manual', 'upload', 'demo', 'sccd']);
export function object(value: unknown, description: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new HttpError(400, `${description} must be an object`);
  return value as Record<string, unknown>;
}
function text(value: unknown, label: string, max: number, required = false): string {
  if (value == null && !required) return '';
  if (typeof value !== 'string' || (required && !value.trim()) || value.length > max || value.includes('\0')) throw new HttpError(400, `${label} must be ${required ? 'non-empty ' : ''}text with at most ${max} characters`);
  return value;
}
function optional(value: unknown, label: string, max: number): string | null {
  const result = text(value, label, max).trim();
  return result || null;
}

export function validateSessions(value: unknown): SessionInput[] {
  if (!Array.isArray(value) || !value.length || value.length > 100) throw new HttpError(400, 'Provide 1–100 conversations per request');
  if (Buffer.byteLength(JSON.stringify(value), 'utf8') > MAX_BYTES) throw new HttpError(413, 'Total input cannot exceed 5 MiB');
  return value.map((raw, si) => {
    const item = object(raw, `Conversation ${si + 1}`);
    const title = text(item.title ?? `Imported conversation ${si + 1}`, 'Title', 200, true).trim();
    const postText = text(item.postText, 'Original post', 10000);
    if (item.includePost !== undefined && typeof item.includePost !== 'boolean') throw new HttpError(400, 'includePost must be true or false');
    const includePost = item.includePost === true;
    if (includePost && !postText.trim()) throw new HttpError(400, 'Provide the original post text when including it in analysis');
    const source = item.source ?? 'upload';
    if (!sources.has(source as Source)) throw new HttpError(400, 'Unsupported conversation source');
    if (!Array.isArray(item.messages) || item.messages.length + Number(includePost) < 1 || item.messages.length + Number(includePost) > 2000) throw new HttpError(400, 'Each conversation requires 1–2000 messages, including the original post if selected');
    const ids = new Set<string>();
    const orders = new Set<number>();
    const messages: MessageInput[] = item.messages.map((rawMessage, mi) => {
      const row = object(rawMessage, `Message ${mi + 1}`);
      const id = text(row.id ?? `m${mi + 1}`, 'Message ID', 128, true).trim();
      if (ids.has(id)) throw new HttpError(400, `Conversation ${title} contains duplicate message ID: ${id}`);
      ids.add(id);
      const order = row.order ?? mi;
      if (typeof order !== 'number' || !Number.isSafeInteger(order) || order < 0) throw new HttpError(400, `Message ${id} order must be a non-negative integer`);
      if (orders.has(order)) throw new HttpError(400, `Conversation ${title} contains duplicate order: ${order}`);
      orders.add(order);
      return { id, text: text(row.text, `Message ${id}`, 10000, true), speaker: optional(row.speaker, 'Speaker', 120), parentId: optional(row.parentId, 'Parent message ID', 128), time: optional(row.time, 'Time', 128), order };
    });
    const parentById = new Map(messages.map(m => [m.id, m.parentId]));
    for (const message of messages) {
      if (message.parentId && !ids.has(message.parentId)) throw new HttpError(400, `Message ${message.id} references a parent outside this conversation: ${message.parentId}`);
      const chain = new Set<string>([message.id]);
      let parent = message.parentId;
      while (parent) {
        if (chain.has(parent)) throw new HttpError(400, `Conversation ${title} contains cyclic reply links`);
        chain.add(parent);
        parent = parentById.get(parent);
      }
    }
    messages.sort((a, b) => a.order! - b.order!);
    return { title, postText, includePost, source: source as Source, messages };
  });
}

export function previewImport(raw: unknown): ImportPreview {
  const request = object(raw, 'Import request');
  if (!['paste', 'csv', 'json'].includes(String(request.format))) throw new HttpError(400, 'Choose paste, CSV or JSON format');
  if (typeof request.content !== 'string' || !request.content.trim()) throw new HttpError(400, 'Paste or upload text to analyze');
  if (Buffer.byteLength(request.content, 'utf8') > MAX_BYTES) throw new HttpError(413, 'The file cannot exceed 5 MiB');
  const content = request.content.replace(/^\uFEFF/, '');
  const warnings: string[] = [];
  let sessions: unknown;
  if (request.format === 'json') {
    try {
      const parsed: unknown = JSON.parse(content);
      sessions = Array.isArray(parsed) ? parsed : parsed && typeof parsed === 'object' && 'sessions' in parsed ? (parsed as { sessions: unknown }).sessions : [parsed];
      warnings.push('Only conversation and message fields are imported. Dataset labels, model scores and other fields are excluded from inference.');
    } catch { throw new HttpError(400, 'Invalid JSON. Check quotes, commas and brackets'); }
  } else if (request.format === 'paste') {
    const lines = content.split(/\r?\n/).filter(line => line.trim());
    const messages = lines.map((line, i) => {
      const match = line.match(/^([^：:\r\n]{1,24})[：:]\s*(.+)$/u);
      const speaker = match && !/^https?$/i.test(match[1].trim()) ? match[1].trim() : null;
      return { id: `m${i + 1}`, text: speaker ? match![2] : line, speaker, order: i };
    });
    sessions = [{ title: request.title || 'Pasted conversation', postText: request.postText, includePost: request.includePost, source: 'manual', messages }];
  } else {
    const parsed = Papa.parse<string[]>(content, { skipEmptyLines: 'greedy' });
    if (parsed.errors.length) throw new HttpError(400, 'Invalid CSV format', parsed.errors.slice(0, 5).map(e => e.message));
    const [header, ...rows] = parsed.data;
    if (!header || rows.length === 0) throw new HttpError(400, 'CSV requires a header and at least one data row');
    const fields = header.map(h => h.trim().toLowerCase());
    if (!fields.includes('text')) throw new HttpError(400, 'CSV is missing the required text column');
    if (new Set(fields).size !== fields.length || fields.some(f => !f)) throw new HttpError(400, 'CSV headers must be distinct and non-empty');
    const known = new Set(['session_id', 'title', 'text', 'id', 'speaker', 'parent_id', 'time', 'order', 'post_text']);
    if (fields.some(f => !known.has(f))) warnings.push('Extra CSV columns, including labels, are ignored during model analysis.');
    const grouped = new Map<string, Record<string, unknown>>();
    rows.forEach((row, ri) => {
      if (row.length !== fields.length) throw new HttpError(400, `CSV row ${ri + 2} has a different column count from the header`);
      const cell = Object.fromEntries(fields.map((field, index) => [field, row[index]]));
      const sid = cell.session_id?.trim() || 'default';
      let group = grouped.get(sid);
      if (!group) {
        group = { title: cell.title?.trim() || request.title || `Imported conversation ${grouped.size + 1}`, postText: cell.post_text || request.postText || '', includePost: request.includePost ?? false, source: 'upload', messages: [] };
        grouped.set(sid, group);
      } else {
        if (cell.title?.trim() && cell.title.trim() !== group.title) throw new HttpError(400, `Conversation ${sid} has inconsistent titles`);
        if (cell.post_text && cell.post_text !== group.postText) throw new HttpError(400, `Conversation ${sid} has inconsistent original posts`);
      }
      const messages = group.messages as Record<string, unknown>[];
      const order = cell.order?.trim();
      messages.push({ id: cell.id?.trim() || `m${messages.length + 1}`, text: cell.text, speaker: cell.speaker, parentId: cell.parent_id, time: cell.time, order: order ? Number(order) : messages.length });
    });
    sessions = [...grouped.values()];
  }
  return { sessions: validateSessions(sessions), warnings };
}
