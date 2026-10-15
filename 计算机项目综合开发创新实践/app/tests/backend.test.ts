import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { EventEmitter, once } from 'node:events';
import { PassThrough, Writable } from 'node:stream';
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { createInterface } from 'node:readline';
import test from 'node:test';
import { previewImport, validateSessions } from '../server/validation.ts';
import { createApp } from '../server/index.ts';
import { Store } from '../server/store.ts';
import { PythonWorker, MODEL_HASH, type InferenceEngine } from '../server/worker.ts';
import { getExamples } from '../server/examples.ts';
import { getBenchmarks } from '../server/benchmarks.ts';
import { APP_ROOT, RESEARCH_ROOT, DEPLOYMENT } from '../server/paths.ts';
import { deploymentIdentity } from '../config/deployment.mjs';
import { predictionScore } from '../shared/types.ts';
import type { Prediction, SessionRecord, WorkerHealth } from '../shared/types.ts';

const input = { title: '课程讨论', source: 'manual', messages: [{ id: 'm1', text: '一起完成小组作业', order: 0 }] };
const mockDeployment = { ...DEPLOYMENT, schemaVersion: 1 as const, modelVersion: 'test-legacy-engine', taskName: 'mock inference only', thresholds: { attack: 0.4 }, conversationThreshold: 0.4 };
const mockIdentity = deploymentIdentity(mockDeployment);
const mockReady = () => ({ type: 'ready', ...mockIdentity, thresholds: mockDeployment.thresholds, conversationThreshold: mockDeployment.conversationThreshold });

test('import preserves Chinese multiline text, grouping and anonymous speakers; strips labels', () => {
  const result = previewImport({ format: 'csv', content: 'session_id,title,id,text,speaker,parent_id,order,label\na,讨论,m1,"第一行\n第二行",同学A,,0,1\na,讨论,m2,我来帮忙,同学B,m1,1,0' });
  assert.equal(result.sessions.length, 1);
  assert.equal(result.sessions[0].messages[0].text, '第一行\n第二行');
  assert.equal(result.sessions[0].messages[1].parentId, 'm1');
  assert.equal(result.sessions[0].messages[0].speaker, '同学A');
  assert.ok(!JSON.stringify(result.sessions).includes('label'));
  assert.equal(previewImport({ format: 'paste', content: '同学A：这周分工\n第二条评论' }).sessions[0].messages[0].text, '这周分工');
  const json = previewImport({ format: 'json', content: JSON.stringify({ sessions: [{ ...input, label: 1, messages: [{ id: 'm1', text: '原文', comment_label: 1 }] }] }) });
  assert.ok(!JSON.stringify(json.sessions).includes('label'));
});

test('invalid uploads are rejected as a whole', () => {
  const invalid = [
    { ...input, messages: [{ id: 'm', text: 'a' }, { id: 'm', text: 'b' }] },
    { ...input, messages: [{ id: 'm', text: '  ' }] },
    { ...input, messages: [{ id: 'm', text: 'a', parentId: 'missing' }] },
    { ...input, messages: [{ id: 'a', text: 'a', parentId: 'b' }, { id: 'b', text: 'b', parentId: 'a' }] },
    { ...input, messages: [{ id: 'm', text: 'a', order: Infinity }] },
    { ...input, includePost: true, postText: '' },
  ];
  for (const value of invalid) assert.throws(() => validateSessions([value]));
  assert.throws(() => previewImport({ format: 'csv', content: 'text,text\na,b' }));
  assert.throws(() => previewImport({ format: 'json', content: '{broken' }));
  assert.throws(() => previewImport({ format: 'paste', content: 'a'.repeat(5 * 1024 * 1024 + 1) }));
  assert.throws(() => validateSessions([input, invalid[0]]));
});

class FakeEngine implements InferenceEngine {
  status: WorkerHealth = { status: 'ready', ...mockIdentity, threshold: 0.4 };
  concurrent = 0;
  maximum = 0;
  fail = false;
  health() { return this.status; }
  async start() { return; }
  async close() { return; }
  async analyze(session: SessionRecord): Promise<Prediction> {
    this.concurrent++;
    this.maximum = Math.max(this.maximum, this.concurrent);
    await new Promise(resolve => setTimeout(resolve, 30));
    this.concurrent--;
    if (this.fail) throw new Error('模拟推理进程中断');
    const message_scores = session.messages.map((m, i) => ({ post_id: session.id, comment_id: m.id, raw_text: m.text, parent_id: m.parentId ?? null, order: m.order ?? i, position: i, comment_time: m.time ?? null, speaker_id: m.speaker ?? null, risk_score: 0.2, risk_probability: 0.2, topic_ids: [], topic_strengths: [], assigned: false, rank: i + 1 }));
    return { post_id: session.id, split: 'inference', label: null, prediction: 0, prediction_name: 'Non-CB', risk_probability: 0.2, threshold: 0.4, topics: [], message_scores, evidence: message_scores.slice(0, 3), coverage: 0, coverage_details: { assigned_messages: 0, total_messages: message_scores.length, fraction: 0, unassigned_message_ids: message_scores.map(m => m.comment_id), zero_topic_messages: message_scores.length } };
  }
}

test('HTTP tasks run one at a time; review, filter, export and deletion persist', async () => {
  const dir = await mkdtemp(path.join(tmpdir(), 'campus-backend-'));
  const engine = new FakeEngine();
  const instance = createApp({ dataDir: dir, engine });
  const server = instance.app.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const address = server.address();
  assert.ok(address && typeof address !== 'string');
  const base = `http://127.0.0.1:${address.port}`;
  async function request(url: string, method = 'GET', body?: unknown) {
    const response = await fetch(base + url, { method, headers: body ? { 'content-type': 'application/json' } : {}, body: body ? JSON.stringify(body) : undefined });
    return { response, body: await response.json() };
  }
  try {
    assert.equal((await request('/api/sessions', 'POST', { sessions: [input, { ...input, messages: [] }] })).response.status, 400);
    assert.equal((await request('/api/health')).body.counts.sessions, 0);
    const created = await request('/api/sessions', 'POST', { sessions: [input, { ...input, title: '第二组', messages: [{ id: 'm1', text: '=危险的表格公式', order: 0 }] }] });
    assert.equal(created.response.status, 201);
    const ids = created.body.sessions.map((s: SessionRecord) => s.id);
    const queued = await request('/api/analyses', 'POST', { sessionIds: ids });
    assert.equal(queued.response.status, 202);
    const analysisIds = queued.body.analyses.map((a: { id: string }) => a.id);
    let detail;
    for (let i = 0; i < 60; i++) {
      detail = (await request(`/api/analyses/${analysisIds[1]}`)).body;
      if (detail.analysis.status === 'succeeded') break;
      await new Promise(resolve => setTimeout(resolve, 10));
    }
    assert.equal(detail.analysis.status, 'succeeded');
    assert.equal(engine.maximum, 1);
    assert.equal((await request('/api/analyses?prediction=0&query=第二')).body.analyses.length, 1);
    assert.equal((await request('/api/analyses?from=not-a-date')).response.status, 400);
    const review = await request(`/api/analyses/${analysisIds[1]}/review`, 'PATCH', { status: 'clear', note: '只是测试文本' });
    assert.equal(review.body.review.note, '只是测试文本');
    assert.equal((await request(`/api/analyses/${analysisIds[1]}/export?format=json`)).body.analysis.review.status, 'clear');
    const csv = await (await fetch(base + `/api/analyses/${analysisIds[1]}/export?format=csv`)).text();
    assert.ok(csv.includes("'=危险的表格公式"));
    assert.ok(csv.includes('只是测试文本'));
    const blocked = await fetch(base + '/api/sessions', { method: 'POST', headers: { origin: 'https://hostile.example', 'content-type': 'application/json' }, body: JSON.stringify({ sessions: [input] }) });
    assert.equal(blocked.status, 403);
    assert.equal((await request(`/api/analyses/${analysisIds[0]}`, 'DELETE')).response.status, 200);
    assert.equal((await request(`/api/analyses/${analysisIds[0]}`)).response.status, 404);
    engine.fail = true;
    const failedJob = (await request('/api/analyses', 'POST', { sessionIds: [ids[1]] })).body.analyses[0];
    for (let i = 0; i < 50; i++) {
      const failed = (await request(`/api/analyses/${failedJob.id}`)).body.analysis;
      if (failed.status === 'failed') { assert.match(failed.error, /中断/); break; }
      if (i === 49) assert.fail('failure was not persisted');
      await new Promise(resolve => setTimeout(resolve, 10));
    }
  } finally {
    await new Promise<void>((resolve, reject) => server.close(e => e ? reject(e) : resolve()));
    await instance.close();
    await rm(dir, { recursive: true, force: true });
  }
});

test('reopening SQLite marks unfinished work interrupted and retains review', async () => {
  const dir = await mkdtemp(path.join(tmpdir(), 'campus-store-'));
  try {
    let store = new Store(dir);
    const session = store.createSessions(validateSessions([input]))[0];
    const analysis = store.createAnalyses([session.id])[0];
    assert.throws(() => store.delete(analysis.id), /结束后才能删除/);
    store.markRunning(analysis.id);
    assert.throws(() => new Store(dir), /另一个运行中的服务/);
    assert.equal(store.detail(analysis.id).analysis.status, 'running');
    store.close();
    store = new Store(dir);
    assert.equal(store.detail(analysis.id).analysis.status, 'failed');
    assert.match(store.detail(analysis.id).analysis.error ?? '', /中断/);
    store.close();
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('actual frozen Python model handles anonymous input and includePost collision without labels', { timeout: 45000 }, async () => {
  const worker = new PythonWorker();
  try {
    await worker.start();
    assert.equal(worker.health().modelHash, MODEL_HASH);
    const session = { ...validateSessions([{ ...input, includePost: true, postText: '大家讨论作业分工', messages: [{ id: '__post__', text: '我可以整理数据', speaker: '同学 A' }] }])[0], id: 'test-real-inference', createdAt: new Date().toISOString(), messageCount: 2 };
    const result = await worker.analyze(session);
    assert.equal(result.label, null);
    assert.equal(result.message_scores.length, 2);
    assert.equal(new Set(result.message_scores.map(m => m.comment_id)).size, 2);
    assert.equal(result.message_scores[1].raw_text, '我可以整理数据');
    assert.ok(Number.isFinite(predictionScore(result)));
  } finally { await worker.close(); }
});

test('missing Python startup becomes an actionable worker error', async () => {
  const worker = new PythonWorker({ pythonPath: path.join(tmpdir(), 'missing-campus-python', 'python.exe'), startupTimeoutMs: 1000 });
  try { await assert.rejects(worker.start()); assert.equal(worker.health().status, 'error'); }
  finally { await worker.close(); }
});

test('local examples validate and frozen six-method benchmarks are complete', async () => {
  const examples = await getExamples();
  assert.equal(examples.filter(e => e.session.source === 'demo').length, 10);
  assert.equal(new Set(examples.map(e => e.id)).size, examples.length);
  assert.ok(examples.some(e => e.id === 'campus-forum-competition'));
  assert.ok(examples.some(e => e.id === 'campus-threat-boundary'));
  const hasSccd = ['sessions.jsonl', 'messages.jsonl'].every(name => existsSync(path.join(RESEARCH_ROOT, 'benchmark', 'data', 'full677', name)));
  const sccdExamples = examples.filter(e => e.session.source === 'sccd');
  if (hasSccd) assert.ok(sccdExamples.length > 0);
  else assert.equal(sccdExamples.length, 0, '交付包不含 SCCD 时应仅保留校园演示样例');
  validateSessions(examples.map(e => e.session));
  assert.ok(sccdExamples.every(e => !JSON.stringify(e.session).includes('label')));
  const benchmark = await getBenchmarks();
  assert.equal(benchmark.methods.length, 6);
  assert.ok(benchmark.methods.every(row => Number.isFinite(row.ap) && row.model_mib > 0));
  assert.match(benchmark.note, /不代表校园场景效果/);
});

test('late events from timed-out worker cannot corrupt its replacement', async () => {
  class Child extends EventEmitter {
    stdout = new PassThrough();
    stderr = new PassThrough();
    stdin = new Writable({ write: (chunk, _encoding, callback) => { this.requests.push(JSON.parse(chunk.toString())); callback(); } });
    requests: { type: string; id: string }[] = [];
    kill() { return true; }
    send(message: unknown) { this.stdout.write(JSON.stringify(message) + '\n'); }
  }
  const children: Child[] = [];
  const worker = new PythonWorker({ deployment: mockDeployment, requestTimeoutMs: 200, spawnProcess: () => {
    const child = new Child();
    children.push(child);
    queueMicrotask(() => child.send(mockReady()));
    return child as unknown as ChildProcessWithoutNullStreams;
  } });
  const session: SessionRecord = { ...validateSessions([input])[0], id: 'generation-test', createdAt: new Date().toISOString(), messageCount: 1 };
  try {
    await assert.rejects(worker.analyze(session), /超时/);
    const next = worker.analyze(session);
    await new Promise(resolve => setTimeout(resolve, 5));
    assert.equal(children.length, 2);
    children[0].stdin.emit('error', new Error('late pipe error'));
    children[0].stdout.write('old non-JSON noise\n');
    children[0].emit('close', 1);
    const result = await new FakeEngine().analyze(session);
    children[1].send({ type: 'result', ...mockIdentity, id: children[1].requests[0].id, result });
    assert.equal(predictionScore(await next), 0.2);
    assert.equal(worker.health().status, 'ready');
  } finally {
    const closing = worker.close();
    children.at(-1)?.emit('close', 0);
    await closing;
  }
});

test('database lock is released after owner process dies', { timeout: 15000 }, async () => {
  const dir = await mkdtemp(path.join(tmpdir(), 'campus-lock-'));
  const program = "import { Store } from './server/store.ts'; const store = new Store(process.env.CAMPUS_TEST_LOCK_DIR); const session = store.createSessions([{title:'interrupted',source:'manual',messages:[{id:'m1',text:'hello'}]}])[0]; const a = store.createAnalyses([session.id])[0]; store.markRunning(a.id); console.log(a.id); setInterval(()=>{},1000);";
  const child = spawn(process.execPath, ['--import', 'tsx', '--input-type=module', '--eval', program], { cwd: APP_ROOT, env: { ...process.env, CAMPUS_TEST_LOCK_DIR: dir }, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
  const reader = createInterface({ input: child.stdout });
  try {
    const [id] = await once(reader, 'line');
    assert.throws(() => new Store(dir), /另一个运行中的服务/);
    const stopped = once(child, 'exit');
    child.kill();
    await stopped;
    const recovered = new Store(dir);
    try { assert.equal(recovered.detail(String(id)).analysis.status, 'failed'); }
    finally { recovered.close(); }
  } finally {
    reader.close();
    if (child.exitCode === null && child.signalCode === null) { const closed = once(child, 'close'); child.kill(); await closed; }
    await rm(dir, { recursive: true, force: true });
  }
});

test('Python exit before ready rejects startup immediately', { timeout: 2000 }, async () => {
  const child = Object.assign(new EventEmitter(), { stdout: new PassThrough(), stderr: new PassThrough(), stdin: new PassThrough(), kill: () => true });
  const worker = new PythonWorker({ startupTimeoutMs: 10000, spawnProcess: () => {
    queueMicrotask(() => child.emit('close', 1));
    return child as unknown as ChildProcessWithoutNullStreams;
  } });
  try {
    await assert.rejects(worker.start(), /进程退出/);
    assert.equal(worker.health().status, 'error');
  } finally { await worker.close(); }
});

test('Python exit after ready rejects active inference immediately', { timeout: 2000 }, async () => {
  const child = Object.assign(new EventEmitter(), { stdout: new PassThrough(), stderr: new PassThrough(), stdin: new PassThrough(), kill: () => true });
  const worker = new PythonWorker({ deployment: mockDeployment, requestTimeoutMs: 10000, spawnProcess: () => {
    queueMicrotask(() => child.stdout.write(JSON.stringify(mockReady()) + '\n'));
    return child as unknown as ChildProcessWithoutNullStreams;
  } });
  try {
    await worker.start();
    const session = { ...validateSessions([input])[0], id: 'exit-active', createdAt: new Date().toISOString(), messageCount: 1 };
    const pending = worker.analyze(session);
    queueMicrotask(() => child.emit('close', 1));
    await assert.rejects(pending, /进程退出/);
    assert.equal(worker.health().status, 'error');
  } finally { await worker.close(); }
});
