import assert from 'node:assert/strict';
import { execFile, fork, type ChildProcess } from 'node:child_process';
import { mkdir, mkdtemp, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { setTimeout as delay } from 'node:timers/promises';
import test from 'node:test';
import Papa from 'papaparse';
import type {
  AnalysisDetail, AnalysisRecord, Health, Prediction, Review, SessionInput, SessionRecord,
} from '../shared/types.ts';
import { isPredictionV2, predictionScore } from '../shared/types.ts';
import { loadDeployment, matchesDeployment } from '../config/deployment.mjs';

const appRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const researchRoot = path.resolve(appRoot, '..', '..', 'cst_mil');
const pythonPath = process.env.CST_PYTHON ?? path.join(researchRoot, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
const deployment = loadDeployment();
const modelPath = deployment.modelAbsolutePath;
const modelHash = deployment.modelSha256;
const base = 'http://127.0.0.1:3292';

function comparablePrediction(value: Prediction): Prediction {
  const copy = structuredClone(value) as Prediction & { inferenceTrace?: Record<string, number> };
  if (copy.inferenceTrace) {
    assert.equal(copy.inferenceTrace.semanticEncodingPasses, 1);
    for (const key of ['encodingSeconds', 'riskPredictionSeconds', 'interpretationSeconds']) {
      assert.ok(Number.isFinite(copy.inferenceTrace[key]) && copy.inferenceTrace[key] >= 0);
      delete copy.inferenceTrace[key];
    }
  }
  return copy;
}

interface ManagedServer {
  child: ChildProcess;
  exited: Promise<void>;
  logs: { stdout: string; stderr: string };
  startupError: Error | null;
  closed: boolean;
}

function startServer(dataDir: string): ManagedServer {
  const child = fork(path.join(appRoot, 'server/index.ts'), [], {
    cwd: appRoot,
    execArgv: ['--import', 'tsx'],
    env: { ...process.env, PORT: '3292', CAMPUS_DATA_DIR: dataDir, CAMPUS_NO_OPEN: '1', CST_PYTHON: pythonPath },
    windowsHide: true,
    stdio: ['ignore', 'pipe', 'pipe', 'ipc'],
  });
  const server: ManagedServer = {
    child, exited: Promise.resolve(), logs: { stdout: '', stderr: '' }, startupError: null, closed: false,
  };
  server.exited = new Promise<void>(resolve => {
    child.once('close', () => { server.closed = true; resolve(); });
  });
  child.on('error', error => { server.startupError = error; });
  child.stdout?.on('data', (chunk: Buffer) => { server.logs.stdout += chunk.toString('utf8'); });
  child.stderr?.on('data', (chunk: Buffer) => { server.logs.stderr += chunk.toString('utf8'); });
  return server;
}

async function waitForReady(server: ManagedServer, signal: AbortSignal): Promise<Health> {
  const deadline = Date.now() + 95_000;
  let lastConnectionError = '';
  while (Date.now() < deadline) {
    signal.throwIfAborted();
    if (server.startupError) throw server.startupError;
    assert.ok(!server.closed, `Server exited before ready.\n${server.logs.stderr}`);
    let health: Health | undefined;
    try {
      const response = await fetch(`${base}/api/health`, {
        signal: AbortSignal.any([signal, AbortSignal.timeout(2_000)]),
      });
      if (response.ok) health = await response.json() as Health;
      else lastConnectionError = `HTTP ${response.status}`;
    } catch (error) {
      lastConnectionError = error instanceof Error ? error.message : String(error);
    }
    if (health?.worker.status === 'error') {
      assert.fail(`Real model startup failed: ${health.worker.error}\n${server.logs.stderr}`);
    }
    if (health?.worker.status === 'ready') {
      assert.equal(health.worker.modelHash, modelHash);
      assert.ok(matchesDeployment(health.worker, deployment));
      return health;
    }
    await delay(200, undefined, { signal });
  }
  assert.fail(`Real model never became ready: ${lastConnectionError}\n${server.logs.stderr}`);
}

async function waitForExit(server: ManagedServer, timeoutMs: number): Promise<boolean> {
  let timer: NodeJS.Timeout | undefined;
  try {
    return await Promise.race([
      server.exited.then(() => true),
      new Promise<boolean>(resolve => { timer = setTimeout(() => resolve(false), timeoutMs); }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

async function stopServer(server: ManagedServer): Promise<boolean> {
  if (server.closed) return true;
  if (server.child.connected) {
    server.child.send('shutdown', error => {
      if (error) server.logs.stderr += `\nShutdown IPC error: ${error.message}`;
    });
  }
  if (await waitForExit(server, 8_000)) return true;
  // Terminate this test's process tree on Windows if graceful shutdown fails.
  if (process.platform === 'win32' && server.child.pid) {
    await new Promise<void>(resolve => {
      execFile('taskkill', ['/PID', String(server.child.pid), '/T', '/F'], { windowsHide: true }, () => resolve());
    });
  } else {
    server.child.kill('SIGKILL');
  }
  assert.ok(await waitForExit(server, 5_000), 'The integration server did not terminate');
  return false;
}

async function request<T>(route: string, signal: AbortSignal, method = 'GET', body?: unknown): Promise<{ response: Response; body: T }> {
  const response = await fetch(base + route, {
    method,
    headers: body === undefined ? undefined : { 'content-type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.any([signal, AbortSignal.timeout(10_000)]),
  });
  return { response, body: await response.json() as T };
}

async function waitForAnalysis(id: string, signal: AbortSignal): Promise<AnalysisDetail> {
  const deadline = Date.now() + 125_000;
  while (Date.now() < deadline) {
    const { response, body } = await request<AnalysisDetail>(`/api/analyses/${id}`, signal);
    assert.equal(response.status, 200);
    assert.notEqual(body.analysis.status, 'failed', body.analysis.error ?? 'Real inference failed');
    if (body.analysis.status === 'succeeded') return body;
    await delay(150, undefined, { signal });
  }
  assert.fail('Real analysis did not finish before its deadline');
}

function predictDirectly(sessions: SessionRecord[], signal: AbortSignal): Promise<Prediction[]> {
  // Deliberately call the research function itself, not the application's worker adapter.
  const program = `
import contextlib, hashlib, importlib, json, sys
from pathlib import Path
protocol = sys.stdout
request = json.load(sys.stdin)
sys.path.insert(0, request["source"])
with contextlib.redirect_stdout(sys.stderr):
    from cst_mil.bench.resources import configure_cpu
    configure_cpu(4)
    import joblib
    module = importlib.import_module(request["deployment"]["predictorModule"])
    predict_session = getattr(module, request["deployment"]["predictFunction"])
    model_path = Path(request["model"])
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == request["modelHash"]
    model = joblib.load(model_path)
    assert isinstance(model, getattr(module, request["deployment"]["modelClass"]))
    if request["deployment"]["modelClass"] == "FollowupPackage":
        module.validate_package(model, request["deployment"], Path(request["source"]).parents[1])
    elif request["deployment"]["schemaVersion"] == 2:
        model.validate_assets()
    results = []
    for session in request["sessions"]:
        messages = [{
            "post_id": session["id"], "comment_id": message["id"], "raw_text": message["text"],
            "order": message.get("order", index), "parent_id": message.get("parentId"),
            "speaker_id": message.get("speaker"), "comment_time": message.get("time"),
        } for index, message in enumerate(session["messages"])]
        if session.get("includePost"):
            existing_ids = {message["comment_id"] for message in messages}
            post_id, suffix = "__post__", 0
            while post_id in existing_ids:
                suffix += 1
                post_id = f"__post___{suffix}"
            messages.insert(0, {
                "post_id": session["id"], "comment_id": post_id, "raw_text": session["postText"],
                "order": min((message["order"] for message in messages), default=0) - 1,
                "parent_id": None, "speaker_id": "原帖", "comment_time": None,
            })
        results.append(predict_session(model, {"post_id": session["id"], "split": "inference"}, messages))
protocol.write(json.dumps(results, ensure_ascii=False, allow_nan=False))
`;
  return new Promise<Prediction[]>((resolve, reject) => {
    const child = execFile(pythonPath, ['-B', '-X', 'utf8', '-c', program], {
      cwd: appRoot,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1' },
      windowsHide: true,
      timeout: 120_000,
      maxBuffer: 16 * 1024 * 1024,
      encoding: 'utf8',
      signal,
    }, (error, stdout, stderr) => {
      if (error) { reject(new Error(`Direct research inference failed: ${error.message}\n${stderr}`)); return; }
      try { resolve(JSON.parse(stdout) as Prediction[]); }
      catch (parseError) { reject(new Error(`Direct inference returned invalid JSON: ${String(parseError)}\n${stderr}`)); }
    });
    child.stdin?.on('error', () => { /* execFile's callback reports a failed process. */ });
    child.stdin?.end(JSON.stringify({ source: path.join(researchRoot, 'src'), model: modelPath, modelHash, sessions, deployment }), 'utf8');
  });
}

test('deployed CST HTTP results equal direct research inference and survive a server restart', { timeout: 360_000 }, async t => {
  const artifactRoot = path.join(appRoot, 'test-results');
  await mkdir(artifactRoot, { recursive: true });
  const dataDir = await mkdtemp(path.join(artifactRoot, 'integration-'));
  const servers: ManagedServer[] = [];
  const input: SessionInput = {
    title: '集成验证：校园课程分工', source: 'manual', includePost: true,
    postText: '课程项目下周答辩，请大家商量分工。',
    messages: [
      { id: 'm1', text: '我负责整理实验结果，今晚发给大家。', speaker: '同学A', order: 0 },
      { id: 'm2', text: '你每次都拖后腿，干脆别来我们组了。', speaker: '同学B', parentId: 'm1', order: 1 },
      { id: 'm3', text: '别这样说，有问题可以一起商量。', speaker: '同学C', parentId: 'm2', order: 2 },
      { id: 'm4', text: '我昨天已经完成表格，可能没有发成功。\n现在重新上传。', speaker: '同学A', parentId: 'm2', order: 3 },
      { id: 'm5', text: '收到了，图表还需要统一格式。', speaker: '同学C', parentId: 'm4', order: 4 },
      { id: 'm6', text: '我们按任务清单来，不要针对个人。', speaker: '同学D', parentId: 'm2', order: 5 },
    ],
  };
  let initialHealth: Health | null = null;
  let firstRequestHealth: Health | null = null;
  try {
    let server = startServer(dataDir);
    servers.push(server);
    initialHealth = await waitForReady(server, t.signal);
    assert.deepEqual(initialHealth.counts, { sessions: 0, analyses: 0 });
    assert.ok(Number.isFinite(initialHealth.worker.startupDurationMs));
    if (deployment.schemaVersion === 2) assert.equal(initialHealth.worker.encoderWarmed, true);
    const contaminated = {
      ...input, label: 1, gold_label: 1,
      messages: input.messages.map(message => ({ ...message, label: 1, comment_label: 1 })),
    };
    const created = await request<{ sessions: SessionRecord[] }>('/api/sessions', t.signal, 'POST', { sessions: [contaminated] });
    assert.equal(created.response.status, 201);
    assert.equal(created.body.sessions.length, 1);
    const session = created.body.sessions[0];
    assert.equal(session.messageCount, input.messages.length + 1);
    assert.ok(!JSON.stringify(session).includes('label'), 'Dataset labels must not enter stored inference inputs');
    const queued = await request<{ analyses: AnalysisRecord[] }>('/api/analyses', t.signal, 'POST', { sessionIds: [session.id] });
    assert.equal(queued.response.status, 202);
    assert.equal(queued.body.analyses.length, 1);
    const id = queued.body.analyses[0].id;
    const detail = await waitForAnalysis(id, t.signal);
    firstRequestHealth = (await request<Health>('/api/health', t.signal)).body;
    assert.ok(Number.isFinite(firstRequestHealth.worker.firstRequestDurationMs));
    assert.equal(firstRequestHealth.worker.firstRequestStatus, 'succeeded');
    assert.ok(detail.result);
    assert.equal(detail.analysis.modelHash, modelHash);
    assert.equal(detail.result.label, null, 'Gold labels must not appear as prediction output labels');
    assert.equal(detail.result.message_scores.length, input.messages.length + 1);

    const [expected] = await predictDirectly([session], t.signal);
    assert.equal(expected.label, null);
    assert.deepEqual<unknown>(comparablePrediction(detail.result), isPredictionV2(expected) ? comparablePrediction(expected) : { ...expected, schemaVersion: 1 }, 'The adapter must preserve every prediction field; only per-request timing metadata differs');
    assert.equal(predictionScore(detail.result), predictionScore(expected));
    if (isPredictionV2(expected)) { assert.ok(isPredictionV2(detail.result)); assert.deepEqual(detail.result.riskCategories, expected.riskCategories); assert.deepEqual(detail.result.conversationRisk, expected.conversationRisk); }
    else { assert.ok(!isPredictionV2(detail.result)); assert.equal(detail.result.threshold, expected.threshold); }
    assert.equal(detail.result.prediction, expected.prediction);
    assert.deepEqual(detail.result.topics, expected.topics);
    assert.deepEqual(detail.result.message_scores, expected.message_scores);
    assert.deepEqual(detail.result.evidence, expected.evidence);
    assert.equal(detail.analysis.riskScore, predictionScore(expected));
    assert.equal(detail.analysis.threshold, isPredictionV2(expected) ? null : expected.threshold);
    assert.equal(detail.analysis.modelVersion, deployment.modelVersion);
    assert.equal(detail.analysis.deploymentHash, deployment.deploymentHash);
    assert.equal(detail.analysis.schemaVersion, deployment.schemaVersion);
    assert.equal(detail.analysis.topicCount, expected.topics.length);

    const note = '已结合上下文复核：保留候选消息，等待进一步核实。\n这是人工备注，不改写模型结果。';
    const reviewed = await request<{ review: Review }>(`/api/analyses/${id}/review`, t.signal, 'PATCH', { status: 'uncertain', note });
    assert.equal(reviewed.response.status, 200);
    assert.equal(reviewed.body.review.note, note);
    assert.equal(reviewed.body.review.status, 'uncertain');
    assert.ok(reviewed.body.review.updatedAt);
    const exported = await request<AnalysisDetail>(`/api/analyses/${id}/export?format=json`, t.signal);
    assert.equal(exported.response.status, 200);
    assert.match(exported.response.headers.get('content-disposition') ?? '', /attachment/i);
    assert.deepEqual(exported.body.result, detail.result);
    assert.equal(exported.body.analysis.review.note, note);
    const csvResponse = await fetch(`${base}/api/analyses/${id}/export?format=csv`, { signal: t.signal });
    assert.equal(csvResponse.status, 200);
    assert.match(csvResponse.headers.get('content-type') ?? '', /text\/csv/i);
    const csv = Papa.parse<Record<string, string>>(await csvResponse.text(), { header: true, skipEmptyLines: true });
    assert.deepEqual(csv.errors, []);
    assert.equal(csv.data.length, detail.result.message_scores.length);
    for (const message of detail.result.message_scores) {
      assert.ok(csv.data.some(row => Object.values(row).includes(message.raw_text)), `CSV lost message ${message.comment_id}`);
    }
    assert.ok(csv.data.some(row => Object.values(row).includes(note)), 'CSV lost the multiline review note');

    assert.equal(await stopServer(server), true, 'Server must honor IPC shutdown before restart');
    server = startServer(dataDir);
    servers.push(server);
    assert.deepEqual((await waitForReady(server, t.signal)).counts, { sessions: 1, analyses: 1 });
    const history = await request<{ analyses: AnalysisRecord[] }>('/api/analyses', t.signal);
    assert.equal(history.response.status, 200);
    assert.equal(history.body.analyses.length, 1);
    assert.equal(history.body.analyses[0].id, id);
    assert.deepEqual(history.body.analyses[0].review, reviewed.body.review);
    const restored = await request<AnalysisDetail>(`/api/analyses/${id}`, t.signal);
    assert.equal(restored.response.status, 200);
    assert.deepEqual(restored.body.result, detail.result);
    assert.deepEqual(restored.body.session, session);
    const deleted = await request<{ ok: boolean }>(`/api/analyses/${id}`, t.signal, 'DELETE');
    assert.equal(deleted.response.status, 200);
    assert.equal(deleted.body.ok, true);
    assert.equal((await request<{ error: string }>(`/api/analyses/${id}`, t.signal)).response.status, 404);
    assert.equal((await request<{ error: string }>(`/api/analyses/${id}/export?format=json`, t.signal)).response.status, 404);
    assert.deepEqual((await request<{ analyses: AnalysisRecord[] }>('/api/analyses', t.signal)).body.analyses, []);
    await writeFile(path.join(dataDir, 'real-model-comparison.json'), JSON.stringify({
      modelHash: initialHealth.worker.modelHash, modelVersion: initialHealth.worker.modelVersion,
      schemaVersion: initialHealth.worker.schemaVersion, taskName: initialHealth.worker.taskName,
      deploymentHash: initialHealth.worker.deploymentHash, manifestPath: deployment.manifestPath,
      trainingSources: deployment.trainingSources, isolatedDataDir: dataDir,
      analysisId: id, score: predictionScore(expected),
      threshold: isPredictionV2(expected) ? null : expected.threshold, topicCount: expected.topics.length,
      messageCount: expected.message_scores.length, exactMatch: true, restartPersistence: true,
      startupToReadyMs: initialHealth.worker.startupDurationMs,
      startupDurationMs: initialHealth.worker.startupDurationMs,
      pythonStartupSeconds: initialHealth.worker.pythonStartupSeconds,
      encoderWarmedBeforeReady: initialHealth.worker.encoderWarmed,
      firstRequestWallMs: firstRequestHealth.worker.firstRequestDurationMs,
      firstRequestDurationMs: firstRequestHealth.worker.firstRequestDurationMs,
      firstRequestStatus: firstRequestHealth.worker.firstRequestStatus,
    }, null, 2), 'utf8');
  } finally {
    for (const [index, server] of servers.entries()) {
      await stopServer(server);
      await writeFile(path.join(dataDir, `server-${index + 1}.log`), `${server.logs.stdout}\nSTDERR\n${server.logs.stderr}`, 'utf8');
    }
  }
});

test('schema2 boundary sessions preserve all text and match direct inference field by field', { skip: deployment.schemaVersion !== 2, timeout: 360_000 }, async t => {
  const dataDir = await mkdtemp(path.join(appRoot, 'test-results', 'integration-boundaries-'));
  const server = startServer(dataDir);
  const cases: SessionInput[] = [
    { title: '单评论', source: 'manual', messages: [{ id: 'm1', text: '请问图书馆周末几点开门？', order: 0 }] },
    { title: '未知词', source: 'manual', messages: [{ id: 'm1', text: 'zzzxqvjk', order: 0 }] },
    { title: '多语言', source: 'manual', messages: [
      { id: 'm1', text: 'Vamos estudar juntos amanhã na biblioteca.', order: 0 },
      { id: 'm2', text: 'We can review the project results together.', parentId: 'm1', order: 1 },
      { id: 'm3', text: '可以，明天下午一起核对实验。', parentId: 'm2', order: 2 },
    ] },
    { title: '混合内容', source: 'manual', messages: [
      { id: 'm1', text: '我们周五检查课程论文里的图表。', order: 0 },
      { id: 'm2', text: '奖学金领取通知：加群后先付199元保证金。', order: 1 },
      { id: 'm3', text: '我整理实验表格，周五带给大家。', parentId: 'm1', order: 2 },
      { id: 'm4', text: '不要付款，请向学校官方渠道核实这条通知。', parentId: 'm2', order: 3 },
    ] },
    { title: '反引用', source: 'manual', messages: [
      { id: 'm1', text: '有人说“你每次都拖后腿”，这种针对个人的说法不合适。', order: 0 },
      { id: 'm2', text: '同意，请停止人身攻击，讨论具体分工。', parentId: 'm1', order: 1 },
    ] },
    { title: '长文本', source: 'manual', messages: [{ id: 'm1', text: '实验结果需要复核，引用资料需标记来源。'.repeat(180) + '\n最终后缀不得截断✅', order: 0 }] },
  ];
  try {
    const health = await waitForReady(server, t.signal);
    assert.equal(health.worker.encoderWarmed, true);
    const created = await request<{ sessions: SessionRecord[] }>('/api/sessions', t.signal, 'POST', { sessions: cases });
    assert.equal(created.response.status, 201);
    assert.equal(created.body.sessions.length, cases.length);
    const queued = await request<{ analyses: AnalysisRecord[] }>('/api/analyses', t.signal, 'POST', { sessionIds: created.body.sessions.map(session => session.id) });
    assert.equal(queued.response.status, 202);
    const actual: AnalysisDetail[] = [];
    for (const analysis of queued.body.analyses) actual.push(await waitForAnalysis(analysis.id, t.signal));
    const expected = await predictDirectly(created.body.sessions, t.signal);
    for (const [index, detail] of actual.entries()) {
      assert.ok(detail.result);
      assert.deepEqual(comparablePrediction(detail.result), comparablePrediction(expected[index]), `${cases[index].title}: HTTP must match every direct prediction field`);
      assert.equal(detail.analysis.modelHash, modelHash);
      assert.equal(detail.analysis.deploymentHash, deployment.deploymentHash);
      assert.equal(detail.analysis.modelVersion, deployment.modelVersion);
      assert.equal(detail.analysis.schemaVersion, 2);
      assert.ok(detail.result && isPredictionV2(detail.result));
      assert.equal(detail.result.riskCategories.length, 3);
      for (const original of created.body.sessions[index].messages) assert.ok(detail.result.message_scores.some(message => message.comment_id === original.id && message.raw_text === original.text));
    }
    const longDetail = actual.at(-1)!;
    const csv = await fetch(`${base}/api/analyses/${longDetail.analysis.id}/export?format=csv`, { signal: t.signal });
    assert.equal(csv.status, 200);
    const parsed = Papa.parse<Record<string, string>>(await csv.text(), { header: true, skipEmptyLines: true });
    assert.deepEqual(parsed.errors, []);
    assert.ok(parsed.data.some(row => Object.values(row).includes(cases.at(-1)!.messages[0].text)), 'CSV must retain the complete long text and suffix');
    const finalHealth = (await request<Health>('/api/health', t.signal)).body;
    await writeFile(path.join(dataDir, 'boundary-case-results.json'), JSON.stringify({
      modelHash, modelVersion: deployment.modelVersion, deploymentHash: deployment.deploymentHash,
      manifestPath: deployment.manifestPath, trainingSources: deployment.trainingSources, isolatedDataDir: dataDir,
      schemaVersion: 2, exactMatch: true, fullTextAndExportPreserved: true,
      startupToReadyMs: health.worker.startupDurationMs, pythonStartupSeconds: health.worker.pythonStartupSeconds,
      startupDurationMs: health.worker.startupDurationMs,
      encoderWarmedBeforeReady: health.worker.encoderWarmed, firstRequestWallMs: finalHealth.worker.firstRequestDurationMs,
      firstRequestDurationMs: finalHealth.worker.firstRequestDurationMs,
      cases: actual.map((detail, index) => ({ name: cases[index].title, session: detail.session, result: detail.result })),
    }, null, 2), 'utf8');
  } finally {
    await stopServer(server);
    await writeFile(path.join(dataDir, 'server.log'), `${server.logs.stdout}\nSTDERR\n${server.logs.stderr}`, 'utf8');
  }
});
