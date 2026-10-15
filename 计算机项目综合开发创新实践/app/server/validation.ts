import Papa from 'papaparse';
import type { ImportPreview, MessageInput, SessionInput, Source } from '../shared/types.ts';

export class HttpError extends Error {
  constructor(public status: number, message: string, public details?: string[]) { super(message); }
}
const MAX_BYTES = 5 * 1024 * 1024;
const sources = new Set<Source>(['manual', 'upload', 'demo', 'sccd']);
export function object(value: unknown, description: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new HttpError(400, `${description}需要一个对象`);
  return value as Record<string, unknown>;
}
function text(value: unknown, label: string, max: number, required = false): string {
  if (value == null && !required) return '';
  if (typeof value !== 'string' || (required && !value.trim()) || value.length > max || value.includes('\0')) throw new HttpError(400, `${label}需为${required ? '非空' : ''}文本，最多 ${max} 字符`);
  return value;
}
function optional(value: unknown, label: string, max: number): string | null {
  const result = text(value, label, max).trim();
  return result || null;
}

export function validateSessions(value: unknown): SessionInput[] {
  if (!Array.isArray(value) || !value.length || value.length > 100) throw new HttpError(400, '每次需要 1–100 个会话');
  if (Buffer.byteLength(JSON.stringify(value), 'utf8') > MAX_BYTES) throw new HttpError(413, '输入总量不能超过 5 MiB');
  return value.map((raw, si) => {
    const item = object(raw, `会话 ${si + 1}`);
    const title = text(item.title ?? `导入会话 ${si + 1}`, '标题', 200, true).trim();
    const postText = text(item.postText, '原帖', 10000);
    if (item.includePost !== undefined && typeof item.includePost !== 'boolean') throw new HttpError(400, 'includePost 必须为 true 或 false');
    const includePost = item.includePost === true;
    if (includePost && !postText.trim()) throw new HttpError(400, '原帖参与分析时，请填写原帖内容');
    const source = item.source ?? 'upload';
    if (!sources.has(source as Source)) throw new HttpError(400, '会话来源不受支持');
    if (!Array.isArray(item.messages) || item.messages.length + Number(includePost) < 1 || item.messages.length + Number(includePost) > 2000) throw new HttpError(400, '每个会话需有 1–2000 条待分析消息（含参与分析的原帖）');
    const ids = new Set<string>();
    const orders = new Set<number>();
    const messages: MessageInput[] = item.messages.map((rawMessage, mi) => {
      const row = object(rawMessage, `第 ${mi + 1} 条消息`);
      const id = text(row.id ?? `m${mi + 1}`, '消息编号', 128, true).trim();
      if (ids.has(id)) throw new HttpError(400, `会话“${title}”存在重复消息编号：${id}`);
      ids.add(id);
      const order = row.order ?? mi;
      if (typeof order !== 'number' || !Number.isSafeInteger(order) || order < 0) throw new HttpError(400, `消息 ${id} 的顺序必须是非负整数`);
      if (orders.has(order)) throw new HttpError(400, `会话“${title}”存在重复顺序：${order}`);
      orders.add(order);
      return { id, text: text(row.text, `消息 ${id}`, 10000, true), speaker: optional(row.speaker, '发言人', 120), parentId: optional(row.parentId, '回复编号', 128), time: optional(row.time, '时间', 128), order };
    });
    const parentById = new Map(messages.map(m => [m.id, m.parentId]));
    for (const message of messages) {
      if (message.parentId && !ids.has(message.parentId)) throw new HttpError(400, `消息 ${message.id} 的回复对象不存在于本会话：${message.parentId}`);
      const chain = new Set<string>([message.id]);
      let parent = message.parentId;
      while (parent) {
        if (chain.has(parent)) throw new HttpError(400, `会话“${title}”存在循环回复关系`);
        chain.add(parent);
        parent = parentById.get(parent);
      }
    }
    messages.sort((a, b) => a.order! - b.order!);
    return { title, postText, includePost, source: source as Source, messages };
  });
}

export function previewImport(raw: unknown): ImportPreview {
  const request = object(raw, '导入请求');
  if (!['paste', 'csv', 'json'].includes(String(request.format))) throw new HttpError(400, '请选择粘贴、CSV 或 JSON 格式');
  if (typeof request.content !== 'string' || !request.content.trim()) throw new HttpError(400, '请输入或上传需要分析的内容');
  if (Buffer.byteLength(request.content, 'utf8') > MAX_BYTES) throw new HttpError(413, '文件不能超过 5 MiB');
  const content = request.content.replace(/^\uFEFF/, '');
  const warnings: string[] = [];
  let sessions: unknown;
  if (request.format === 'json') {
    try {
      const parsed: unknown = JSON.parse(content);
      sessions = Array.isArray(parsed) ? parsed : parsed && typeof parsed === 'object' && 'sessions' in parsed ? (parsed as { sessions: unknown }).sessions : [parsed];
      warnings.push('仅导入会话和消息字段；数据集标签、模型分数等额外字段不会用于推理。');
    } catch { throw new HttpError(400, 'JSON 格式不正确，请检查引号、逗号和括号'); }
  } else if (request.format === 'paste') {
    const lines = content.split(/\r?\n/).filter(line => line.trim());
    const messages = lines.map((line, i) => {
      const match = line.match(/^([^：:\r\n]{1,24})[：:]\s*(.+)$/u);
      const speaker = match && !/^https?$/i.test(match[1].trim()) ? match[1].trim() : null;
      return { id: `m${i + 1}`, text: speaker ? match![2] : line, speaker, order: i };
    });
    sessions = [{ title: request.title || '粘贴会话', postText: request.postText, includePost: request.includePost, source: 'manual', messages }];
  } else {
    const parsed = Papa.parse<string[]>(content, { skipEmptyLines: 'greedy' });
    if (parsed.errors.length) throw new HttpError(400, 'CSV 格式不正确', parsed.errors.slice(0, 5).map(e => e.message));
    const [header, ...rows] = parsed.data;
    if (!header || rows.length === 0) throw new HttpError(400, 'CSV 需要表头和至少一条数据');
    const fields = header.map(h => h.trim().toLowerCase());
    if (!fields.includes('text')) throw new HttpError(400, 'CSV 缺少必需的 text 列');
    if (new Set(fields).size !== fields.length || fields.some(f => !f)) throw new HttpError(400, 'CSV 表头不能重复或为空');
    const known = new Set(['session_id', 'title', 'text', 'id', 'speaker', 'parent_id', 'time', 'order', 'post_text']);
    if (fields.some(f => !known.has(f))) warnings.push('CSV 的额外列（包括标签）已忽略，不参与模型分析。');
    const grouped = new Map<string, Record<string, unknown>>();
    rows.forEach((row, ri) => {
      if (row.length !== fields.length) throw new HttpError(400, `CSV 第 ${ri + 2} 行的列数与表头不一致`);
      const cell = Object.fromEntries(fields.map((field, index) => [field, row[index]]));
      const sid = cell.session_id?.trim() || 'default';
      let group = grouped.get(sid);
      if (!group) {
        group = { title: cell.title?.trim() || request.title || `导入会话 ${grouped.size + 1}`, postText: cell.post_text || request.postText || '', includePost: request.includePost ?? false, source: 'upload', messages: [] };
        grouped.set(sid, group);
      } else {
        if (cell.title?.trim() && cell.title.trim() !== group.title) throw new HttpError(400, `会话 ${sid} 的标题不一致`);
        if (cell.post_text && cell.post_text !== group.postText) throw new HttpError(400, `会话 ${sid} 的原帖不一致`);
      }
      const messages = group.messages as Record<string, unknown>[];
      const order = cell.order?.trim();
      messages.push({ id: cell.id?.trim() || `m${messages.length + 1}`, text: cell.text, speaker: cell.speaker, parentId: cell.parent_id, time: cell.time, order: order ? Number(order) : messages.length });
    });
    sessions = [...grouped.values()];
  }
  return { sessions: validateSessions(sessions), warnings };
}
