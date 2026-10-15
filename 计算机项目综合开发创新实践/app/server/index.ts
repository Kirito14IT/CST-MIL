import express, { type ErrorRequestHandler } from 'express';
import Papa from 'papaparse';
import path from 'node:path';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import type { ReviewStatus, AnalysisDetail } from '../shared/types.ts';
import { isPredictionV2 } from '../shared/types.ts';
import { APP_ROOT } from './paths.ts';
import { HttpError, object, previewImport, validateSessions } from './validation.ts';
import { Store, type Filters } from './store.ts';
import { PythonWorker, type InferenceEngine } from './worker.ts';
import { AnalysisQueue } from './queue.ts';
import { getExamples } from './examples.ts';
import { getBenchmarks } from './benchmarks.ts';

interface AppOptions { dataDir?: string; engine?: InferenceEngine }
function queryString(value: unknown, name: string): string | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== 'string') throw new HttpError(400, `${name} 参数格式错误`);
  return value;
}
function dateFilter(value: unknown, name: string) {
  const date = queryString(value, name);
  if (date !== undefined && (!/^\d{4}-\d{2}-\d{2}$/.test(date) || Number.isNaN(Date.parse(date + 'T00:00:00Z')) || new Date(date + 'T00:00:00Z').toISOString().slice(0, 10) !== date)) throw new HttpError(400, `${name} 必须是有效的 YYYY-MM-DD 日期`);
  return date;
}
function csvSafe(value: unknown): string | number | null {
  if (typeof value === 'number') return value;
  if (value === null || value === undefined) return null;
  const string = String(value);
  return /^[\s]*[=+\-@]|^[\t\r\n]/.test(string) ? "'" + string : string;
}
function csvExport(detail: AnalysisDetail) {
  const { analysis, result, session } = detail;
  const messages = result?.message_scores ?? session.messages.map((m, i) => ({ comment_id: m.id, speaker_id: m.speaker, raw_text: m.text, order: m.order ?? i, parent_id: m.parentId, comment_time: m.time, risk_score: null, topic_ids: [] as number[] }));
  const rows = messages.map(message => Object.fromEntries(Object.entries({
    analysis_id: analysis.id, session_id: session.id, title: session.title, source: session.source, status: analysis.status,
    created_at: analysis.createdAt, model_version: analysis.modelVersion, model_sha256: analysis.modelHash,
    schema_version: analysis.schemaVersion, deployment_sha256: analysis.deploymentHash, score_type: analysis.scoreType,
    session_risk_score: analysis.riskScore, threshold: analysis.threshold, prediction: analysis.prediction,
    overall_evidence_score: result && isPredictionV2(result) ? result.overallEvidenceScore : null,
    conversation_reference_score: result && isPredictionV2(result) ? result.conversationRisk.score : null,
    conversation_reference_domain: result && isPredictionV2(result) ? result.conversationRisk.domain : null,
    detected_risk_categories: result && isPredictionV2(result) ? result.riskCategories.filter(category => category.detected).map(category => category.id).join(';') : null,
    message_category_scores: 'riskCategoryScores' in message && message.riskCategoryScores ? JSON.stringify(message.riskCategoryScores) : null,
    message_id: message.comment_id, order: message.order, speaker: message.speaker_id, parent_id: message.parent_id,
    time: message.comment_time, text: message.raw_text, message_risk_score: message.risk_score,
    topic_ids: message.topic_ids.join(';'), topic_titles: (result?.topics ?? []).filter(t => message.topic_ids.includes(t.topic_id)).map(t => t.title).join(';'),
    is_candidate: result?.evidence.some(e => e.comment_id === message.comment_id) ? 'yes' : 'no',
    review_status: analysis.review.status, review_note: analysis.review.note,
  }).map(([key, value]) => [key, csvSafe(value)])));
  return '\uFEFF' + Papa.unparse(rows, { newline: '\r\n' });
}

export function createApp(options: AppOptions = {}) {
  const store = new Store(options.dataDir ?? process.env.CAMPUS_DATA_DIR ?? path.join(APP_ROOT, 'data'));
  const engine = options.engine ?? new PythonWorker();
  const queue = new AnalysisQueue(store, engine);
  const app = express();
  let closed = false;
  app.disable('x-powered-by');
  app.use((req, res, next) => {
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('Referrer-Policy', 'no-referrer');
    if (req.path.startsWith('/api/')) res.setHeader('Cache-Control', 'no-store');
    if (closed) return next(new HttpError(503, '服务正在关闭'));
    if (!['127.0.0.1', 'localhost', '[::1]'].includes(req.hostname)) return next(new HttpError(403, '仅允许本机访问'));
    const origin = req.headers.origin;
    if (origin) {
      try { if (!['127.0.0.1', 'localhost', '[::1]'].includes(new URL(origin).hostname)) return next(new HttpError(403, '不接受外部网页的请求')); }
      catch { return next(new HttpError(403, '请求来源无效')); }
    }
    if (!['GET', 'HEAD', 'OPTIONS'].includes(req.method) && req.headers['sec-fetch-site'] === 'cross-site') return next(new HttpError(403, '不接受跨站写入请求'));
    if (['POST', 'PATCH'].includes(req.method) && !req.is('application/json')) return next(new HttpError(415, '请使用 application/json 请求格式'));
    next();
  });
  app.use(express.json({ limit: '6mb', strict: true }));
  app.get('/api/health', (_req, res) => res.json({ app: 'Campus Text Lab', appProtocolVersion: 2, worker: engine.health(), counts: store.counts() }));
  app.get('/api/examples', async (_req, res) => res.json({ examples: await getExamples() }));
  app.post('/api/import/preview', (req, res) => res.json(previewImport(req.body)));
  app.post('/api/sessions', (req, res) => {
    const body = object(req.body, '请求');
    res.status(201).json({ sessions: store.createSessions(validateSessions(body.sessions)) });
  });
  app.post('/api/analyses', (req, res) => {
    const body = object(req.body, '请求');
    if (!Array.isArray(body.sessionIds) || !body.sessionIds.length || body.sessionIds.length > 100 || body.sessionIds.some(id => typeof id !== 'string' || id.length > 128) || new Set(body.sessionIds).size !== body.sessionIds.length) throw new HttpError(400, '请选择 1–100 个不重复的有效会话编号');
    const analyses = store.createAnalyses(body.sessionIds as string[]);
    queue.enqueue(analyses.map(row => row.id));
    res.status(202).json({ analyses });
  });
  app.get('/api/analyses', (req, res) => {
    const query = queryString(req.query.query, 'query');
    if (query && query.length > 200) throw new HttpError(400, '查询词最多 200 字符');
    const prediction = queryString(req.query.prediction, 'prediction');
    if (prediction !== undefined && !['0', '1'].includes(prediction)) throw new HttpError(400, 'prediction 只能是 0 或 1');
    const filters: Filters = { query, from: dateFilter(req.query.from, 'from'), to: dateFilter(req.query.to, 'to'), prediction: prediction === undefined ? undefined : Number(prediction) };
    if (filters.from && filters.to && filters.from > filters.to) throw new HttpError(400, '开始日期不能晚于结束日期');
    res.json({ analyses: store.list(filters) });
  });
  app.get('/api/analyses/:id', (req, res) => res.json(store.detail(String(req.params.id))));
  app.patch('/api/analyses/:id/review', (req, res) => {
    const body = object(req.body, '核查意见');
    if (!['pending', 'risk', 'clear', 'uncertain'].includes(String(body.status)) || typeof body.note !== 'string' || body.note.length > 5000 || body.note.includes('\0')) throw new HttpError(400, '核查状态无效，或备注超过 5000 字符');
    res.json({ review: store.review(String(req.params.id), body.status as ReviewStatus, body.note) });
  });
  app.delete('/api/analyses/:id', (req, res) => { store.delete(String(req.params.id)); res.json({ ok: true }); });
  app.get('/api/analyses/:id/export', (req, res) => {
    const format = queryString(req.query.format, 'format') ?? 'json';
    if (!['json', 'csv'].includes(format)) throw new HttpError(400, '仅支持 JSON 或 CSV 导出');
    const detail = store.detail(String(req.params.id));
    res.setHeader('Content-Disposition', `attachment; filename="analysis-${detail.analysis.id}.${format}"`);
    if (format === 'json') res.type('application/json; charset=utf-8').send(JSON.stringify(detail, null, 2));
    else res.type('text/csv; charset=utf-8').send(csvExport(detail));
  });
  app.get('/api/benchmarks', async (_req, res) => res.json(await getBenchmarks()));
  app.use('/api', (_req, _res, next) => next(new HttpError(404, '接口不存在')));
  const dist = path.join(APP_ROOT, 'dist');
  app.use(express.static(dist));
  app.use((req, res, next) => {
    if (req.method === 'GET' && existsSync(path.join(dist, 'index.html'))) res.sendFile(path.join(dist, 'index.html'));
    else next(new HttpError(404, '页面未构建，请在项目目录运行 npm start'));
  });
  const errors: ErrorRequestHandler = (error: unknown, _req, res, _next) => {
    if (error instanceof HttpError) { res.status(error.status).json({ error: error.message, ...(error.details ? { details: error.details } : {}) }); return; }
    if (error && typeof error === 'object' && 'type' in error && error.type === 'entity.too.large') { res.status(413).json({ error: '请求体过大，输入内容限 5 MiB' }); return; }
    if (error instanceof SyntaxError) { res.status(400).json({ error: '请求 JSON 格式错误' }); return; }
    console.error('[API]', error instanceof Error ? error.message : '未知错误');
    res.status(500).json({ error: '服务处理失败，请查看终端并重试' });
  };
  app.use(errors);
  void engine.start().catch(() => { /* Startup details are exposed through /api/health. */ });
  return { app, store, engine, async close() { if (closed) return; closed = true; await queue.close(); store.close(); } };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const port = Number(process.env.PORT ?? 3210);
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('PORT 必须在 1–65535 之间');
  const instance = createApp();
  const server = instance.app.listen(port, '127.0.0.1', () => console.log(`Campus Text Lab: http://127.0.0.1:${port}`));
  let stopping = false;
  const shutdown = async () => {
    if (stopping) return;
    stopping = true;
    server.close();
    server.closeIdleConnections();
    await instance.close();
    server.closeAllConnections();
    process.exit(0);
  };
  process.on('message', message => { if (message === 'shutdown') void shutdown(); });
  process.on('SIGINT', () => { void shutdown(); });
  process.on('SIGTERM', () => { void shutdown(); });
  server.on('error', error => { console.error(error.message); void instance.close().finally(() => process.exit(1)); });
}
