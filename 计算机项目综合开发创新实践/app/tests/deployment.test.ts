import assert from 'node:assert/strict';
import test from 'node:test';
import { DatabaseSync } from 'node:sqlite';
import { mkdtemp, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { EventEmitter, once } from 'node:events';
import { PassThrough } from 'node:stream';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import type { ChildProcessWithoutNullStreams } from 'node:child_process';
import { DEPLOYMENT } from '../server/paths.ts';
import { loadDeployment, deploymentIdentity, matchesDeployment, resolveRepoPath, assertReusableService, type LoadedDeployment } from '../config/deployment.mjs';
import { validatePrediction } from '../server/prediction.ts';
import { PythonWorker } from '../server/worker.ts';
import { Store } from '../server/store.ts';
import { createApp } from '../server/index.ts';
import type { PredictionV1, PredictionV2, SessionRecord } from '../shared/types.ts';

const identity = deploymentIdentity(DEPLOYMENT);
const session: SessionRecord = { id: 's', title: '独立测试', source: 'manual', createdAt: '2026-10-05', messageCount: 1, messages: [{ id: 'm', text: '保留完整原文' }] };
const location = { post_id: 's', comment_id: 'm', raw_text: '保留完整原文', parent_id: null, order: 0, position: 0, comment_time: null, speaker_id: null };
const legacy: PredictionV1 = { post_id: 's', split: 'inference', label: null, prediction: 0, prediction_name: 'Non-CB', risk_probability: 0.2, threshold: 0.4, topics: [], message_scores: [{ ...location, risk_score: 0.2, topic_ids: [], topic_strengths: [], assigned: false, rank: 1 }], evidence: [{ ...location, risk_score: 0.2 }], coverage: 0, coverage_details: { assigned_messages: 0, total_messages: 1, fraction: 0, unassigned_message_ids: ['m'], zero_topic_messages: 1 } };
const v2Deployment: LoadedDeployment = { ...DEPLOYMENT, schemaVersion: 2, modelVersion: 'test-v2', taskName: 'three category fixture', predictorModule: 'cst_mil.unified_semantic_v1_1.model', modelClass: 'UnifiedSemanticModel', thresholds: { attack: 0.5, hate: 0.6, spam_scam: 0.7 }, conversationThreshold: 0.45 };
const fixture = (): PredictionV2 => ({ post_id: 's', split: 'inference', label: null, schemaVersion: 2, modelVersion: 'test-v2', prediction: 1, prediction_name: '风险线索', overallEvidenceScore: 0.8, scoreType: 'max_message_category_score', topics: [], message_scores: [{ ...location, risk_score: 0.8, riskCategoryScores: { attack: 0.8, hate: 0.2, spam_scam: 0.1 }, topic_ids: [], topic_strengths: [], assigned: false, rank: 1 }], evidence: [{ ...location, risk_score: 0.8 }], riskCategories: [
  { id: 'attack', name: '攻击性', score: 0.8, threshold: 0.5, detected: true, message_ids: ['m'], evidence: [{ ...location, risk_score: 0.8 }] },
  { id: 'hate', name: '仇恨', score: 0.2, threshold: 0.6, detected: false, message_ids: [], evidence: [] },
  { id: 'spam_scam', name: '垃圾或诈骗', score: 0.1, threshold: 0.7, detected: false, message_ids: [], evidence: [] },
], conversationRisk: { score: 0.3, threshold: 0.45, prediction: 0, domain: 'SCCD', note: '会话迁移参考' }, coverage: 0, coverage_details: { assigned_messages: 0, total_messages: 1, fraction: 0, unassigned_message_ids: ['m'], zero_topic_messages: 1 } });

const groundedFixture = (): PredictionV2 => {
  const result = fixture();
  const raw = '😀 Cafe\u0301 Straße ΟΣ';
  for (const message of [...result.message_scores, ...result.evidence, ...result.riskCategories.flatMap(category => category.evidence)]) message.raw_text = raw;
  result.message_scores[0].assigned = true;
  result.message_scores[0].topic_ids = [0]; result.message_scores[0].topic_strengths = [1];
  result.topics = [{
    topic_id: 0, title: 'café / strasse', keywords: ['café', 'strasse', 'οσ'], topic_strength: 1, risk_score: 0.8,
    message_ids: ['m'], support_messages: [{ ...location, raw_text: raw }], segments: [], evidence: [{ ...location, raw_text: raw, risk_score: 0.8 }],
    keyword_supports: {
      café: [{ message_id: 'm', start: 2, end: 7, excerpt: 'Cafe\u0301' }],
      strasse: [{ message_id: 'm', start: 8, end: 14, excerpt: 'Straße' }],
      οσ: [{ message_id: 'm', start: 15, end: 17, excerpt: 'ΟΣ' }],
    },
  }];
  result.coverage = 1; result.coverage_details = { assigned_messages: 1, total_messages: 1, fraction: 1, unassigned_message_ids: [], zero_topic_messages: 0 };
  return result;
};

test('deployment identity comes from the manifest and rejects path escape, unknown imports and changed assets', async () => {
  assert.ok(matchesDeployment(identity, DEPLOYMENT));
  assert.ok(!matchesDeployment({ ...identity, deploymentHash: '0'.repeat(64) }, DEPLOYMENT));
  assert.ok(!matchesDeployment({ modelHash: identity.modelHash }, DEPLOYMENT), 'Old health responses must not silently match');
  assert.throws(() => resolveRepoPath('../outside'), /越出/);
  assert.throws(() => resolveRepoPath(path.resolve('outside')), /相对/);
  const dir = await mkdtemp(path.join(tmpdir(), 'campus-manifest-'));
  const file = path.join(dir, 'deployment.json');
  try {
    for (const change of [{ predictorModule: 'user_supplied.module' }, { schemaVersion: 2, predictorModule: 'cst_mil.unified_semantic.model', modelClass: 'UnifiedSemanticModel' }, { modelSha256: '0'.repeat(64) }, { codeFingerprints: { [Object.keys(DEPLOYMENT.codeFingerprints)[0]]: '0'.repeat(64) } }]) {
      await writeFile(file, JSON.stringify({ ...DEPLOYMENT, ...change }));
      assert.throws(() => loadDeployment(file));
    }
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('schema2 validates separate category and SCCD reference scores, without legacy probability', () => {
  assert.equal(validatePrediction(fixture(), v2Deployment, 's').schemaVersion, 2);
  const mutate = [
    (r: PredictionV2) => { r.overallEvidenceScore = 0.7; },
    (r: PredictionV2) => { r.riskCategories[0].threshold = 0.4; },
    (r: PredictionV2) => { r.riskCategories[0].message_ids = ['invented']; },
    (r: PredictionV2) => { r.riskCategories[0].evidence[0].raw_text = '伪造摘录'; },
    (r: PredictionV2) => { r.message_scores[0].riskCategoryScores!.hate = NaN; },
    (r: PredictionV2) => { r.conversationRisk.prediction = 1; },
    (r: PredictionV2) => { r.modelVersion = 'old'; },
  ];
  for (const change of mutate) { const result = fixture(); change(result); assert.throws(() => validatePrediction(result, v2Deployment, 's')); }
  assert.throws(() => validatePrediction({ ...fixture(), risk_probability: 0.8 }, v2Deployment, 's'));
  assert.throws(() => validatePrediction({ ...fixture(), topics: [null] }, v2Deployment, 's'));
  assert.throws(() => validatePrediction({ ...fixture(), coverage_details: undefined }, v2Deployment, 's'));
  assert.throws(() => validatePrediction({ ...fixture(), coverage_details: { ...fixture().coverage_details, assigned_messages: 1 } }, v2Deployment, 's'));
  const badMessageScore = fixture(); badMessageScore.message_scores[0].risk_score = 0; badMessageScore.evidence[0].risk_score = 0; badMessageScore.riskCategories[0].evidence[0].risk_score = 0.8;
  assert.throws(() => validatePrediction(badMessageScore, v2Deployment, 's'), /最大值/);
  const badEvidenceScore = fixture(); badEvidenceScore.evidence[0].risk_score = 1;
  assert.throws(() => validatePrediction(badEvidenceScore, v2Deployment, 's'), /分数/);
  const invented = fixture();
  for (const message of [...invented.message_scores, ...invented.evidence, ...invented.riskCategories.flatMap(category => category.evidence)]) message.raw_text = '一致但不在输入中的新文本';
  assert.throws(() => validatePrediction(invented, v2Deployment, 's', session), /输入消息/);
});

test('launcher reuse gate rejects old service and different deployment, while allowing an exact ready identity', () => {
  assert.throws(() => assertReusableService({ app: 'Campus Text Lab', worker: { status: 'ready', modelHash: identity.modelHash } }, DEPLOYMENT), /旧模型/);
  assert.throws(() => assertReusableService({ app: 'Campus Text Lab', appProtocolVersion: 2, worker: { status: 'ready', ...identity, modelVersion: 'different' } }, DEPLOYMENT), /旧模型/);
  assert.doesNotThrow(() => assertReusableService({ app: 'Campus Text Lab', appProtocolVersion: 2, worker: { status: 'ready', ...identity } }, DEPLOYMENT));
});

test('schema2 topic supports use original Unicode codepoint offsets and reject forged names or excerpts', () => {
  const valid = groundedFixture();
  assert.doesNotThrow(() => validatePrediction(valid, v2Deployment, 's'));
  const changes = [
    (r: PredictionV2) => { r.topics[0].title = '臆造主题'; },
    (r: PredictionV2) => { r.topics[0].keyword_supports!.café[0].excerpt = '伪造摘录'; },
    (r: PredictionV2) => { r.topics[0].keyword_supports!.café[0].start = 3; r.topics[0].keyword_supports!.café[0].end = 8; }, // UTF-16 offsets after emoji are wrong here.
    (r: PredictionV2) => { r.topics[0].keyword_supports!.café[0].start = -1; },
    (r: PredictionV2) => { r.topics[0].keyword_supports!.café = []; },
    (r: PredictionV2) => { delete r.topics[0].keyword_supports; },
    (r: PredictionV2) => { r.topics[0].keyword_supports!.invented = []; },
    (r: PredictionV2) => { r.topics[0].keyword_supports!.café[0].message_id = 'invented'; },
  ];
  for (const change of changes) { const bad = groundedFixture(); change(bad); assert.throws(() => validatePrediction(bad, v2Deployment, 's')); }
  const outside = groundedFixture();
  outside.message_scores.push({ ...outside.message_scores[0], comment_id: 'outside', assigned: false, topic_ids: [], topic_strengths: [], rank: 2 });
  outside.coverage = 0.5; outside.coverage_details = { assigned_messages: 1, total_messages: 2, fraction: 0.5, unassigned_message_ids: ['outside'], zero_topic_messages: 1 };
  outside.topics[0].keyword_supports!.café[0].message_id = 'outside';
  assert.throws(() => validatePrediction(outside, v2Deployment, 's'), /不属于本话题/);
  assert.doesNotThrow(() => validatePrediction({ ...legacy, schemaVersion: 1 }, { ...DEPLOYMENT, thresholds: { attack: 0.4 } }, 's'), 'Legacy results do not need new keyword supports');
});

test('Python schema2 grounding verifies NFC casefold, exact excerpts and membership without loading a model', async () => {
  const python = process.env.CST_PYTHON ?? path.resolve('../..', 'cst_mil/.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
  const valid = groundedFixture();
  const messages = [{ comment_id: 'm', raw_text: valid.message_scores[0].raw_text }, { comment_id: 'outside', raw_text: valid.message_scores[0].raw_text }];
  const mutants = [
    (r: PredictionV2) => { r.topics[0].title = '臆造主题'; },
    (r: PredictionV2) => { r.topics[0].keyword_supports!.café[0].excerpt = 'café'; },
    (r: PredictionV2) => { r.topics[0].keyword_supports!.café[0].start = 3; r.topics[0].keyword_supports!.café[0].end = 8; },
    (r: PredictionV2) => { r.topics[0].keyword_supports!.café[0].message_id = 'outside'; },
    (r: PredictionV2) => { r.topics[0].keyword_supports!.café[0].start = true as unknown as number; },
    (r: PredictionV2) => { r.topics[0].keywords[0] = 'invented'; r.topics[0].title = 'invented / strasse'; r.topics[0].keyword_supports!.invented = r.topics[0].keyword_supports!.café; delete r.topics[0].keyword_supports!.café; },
    (r: PredictionV2) => { r.topics[0].support_messages[0].raw_text = '一致但并非原文'; },
  ].map(change => { const bad = groundedFixture(); change(bad); return bad; });
  const generic = groundedFixture();
  generic.topics[0].title = 'ｖｅｒｙ'; generic.topics[0].keywords = ['ｖｅｒｙ'];
  generic.topics[0].support_messages[0].raw_text = '😀 ＶＥＲＹ';
  generic.topics[0].keyword_supports = { 'ｖｅｒｙ': [{ message_id: 'm', start: 2, end: 6, excerpt: 'ＶＥＲＹ' }] };
  const program = `import importlib.util,json,pathlib,sys,unicodedata
spec=importlib.util.spec_from_file_location('campus_worker', pathlib.Path('python/worker.py'))
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
data=json.loads(sys.argv[1])
is_generic=lambda word: unicodedata.normalize('NFKC',word).casefold() in {'very','晚上','很'}
module.validate_topic_grounding(data['valid'],data['messages'],is_generic)
module.validate_topic_grounding({'topics':[]},data['messages'],is_generic)
rejected=0
for result in data['mutants']:
    try: module.validate_topic_grounding(result,data['messages'],is_generic)
    except ValueError: rejected+=1
    else: raise AssertionError('forged topic accepted')
try: module.validate_topic_grounding(data['generic'],[{'comment_id':'m','raw_text':'😀 ＶＥＲＹ'}],is_generic)
except ValueError: rejected+=1
else: raise AssertionError('generic-only name accepted')
assert rejected==len(data['mutants'])+1
assert 'torch' not in sys.modules and 'joblib' not in sys.modules
print('grounding-verified-without-model')`;
  const output = await promisify(execFile)(python, ['-B', '-X', 'utf8', '-c', program, JSON.stringify({ valid, messages, mutants, generic })], { windowsHide: true });
  assert.match(output.stdout, /grounding-verified-without-model/);
});

test('a worker with mismatched deployment handshake is rejected before inference', async () => {
  const child = Object.assign(new EventEmitter(), { stdout: new PassThrough(), stderr: new PassThrough(), stdin: new PassThrough(), kill: () => true });
  const worker = new PythonWorker({ startupTimeoutMs: 1000, spawnProcess: () => {
    queueMicrotask(() => child.stdout.write(JSON.stringify({ type: 'ready', ...identity, deploymentHash: '0'.repeat(64), thresholds: DEPLOYMENT.thresholds, conversationThreshold: DEPLOYMENT.conversationThreshold }) + '\n'));
    return child as unknown as ChildProcessWithoutNullStreams;
  } });
  try { await assert.rejects(worker.start(), /身份/); assert.equal(worker.health().status, 'error'); }
  finally { const closing = worker.close(); child.emit('close', 0); await closing; }
});

test('schema migration backs up and preserves 18 legacy results and notes, then stores actual v2 identity', async () => {
  const dir = await mkdtemp(path.join(tmpdir(), 'campus-migration-'));
  const filename = path.join(dir, 'campus.sqlite');
  const old = new DatabaseSync(filename);
  old.exec(`CREATE TABLE sessions(id TEXT PRIMARY KEY,body TEXT NOT NULL); CREATE TABLE analyses(id TEXT PRIMARY KEY,session_id TEXT NOT NULL,title TEXT NOT NULL,source TEXT NOT NULL,message_count INTEGER NOT NULL,status TEXT NOT NULL,created_at TEXT NOT NULL,completed_at TEXT,prediction INTEGER,risk_score REAL,topic_count INTEGER,error TEXT,model_hash TEXT,model_version TEXT NOT NULL,threshold REAL,duration_ms REAL,result TEXT,review_status TEXT NOT NULL DEFAULT 'pending',review_note TEXT NOT NULL DEFAULT '',review_updated_at TEXT); PRAGMA user_version=1;`);
  const insert = old.prepare('INSERT INTO analyses(id,session_id,title,source,message_count,status,created_at,prediction,risk_score,topic_count,model_hash,model_version,threshold,result,review_status,review_note) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)');
  for (let index = 0; index < 18; index++) {
    const id = String(index);
    old.prepare('INSERT INTO sessions(id,body) VALUES(?,?)').run(id, JSON.stringify({ ...session, id }));
    insert.run(id, id, `旧历史${index}`, 'manual', 1, 'succeeded', '2026-10-04', 0, 0.2, 0, 'old-checkpoint', '旧版本', 0.4, JSON.stringify(legacy), 'uncertain', `原核查备注${index}\n保留上下文`);
  }
  const before = old.prepare('SELECT * FROM analyses ORDER BY id').all();
  old.close();
  const store = new Store(dir);
  try {
    assert.equal(store.list().length, 18);
    for (const row of before) { const detail = store.detail(String(row.id)); assert.equal(detail.analysis.modelVersion, row.model_version); assert.equal(detail.analysis.review.note, row.review_note); assert.deepEqual(detail.result, JSON.parse(String(row.result))); assert.equal(detail.analysis.schemaVersion, 1); }
    const backups = await readdir(path.join(dir, 'backups'));
    assert.equal(backups.length, 1);
    const backup = new DatabaseSync(path.join(dir, 'backups', backups[0]), { readOnly: true });
    try { assert.deepEqual(backup.prepare('SELECT * FROM analyses ORDER BY id').all(), before); } finally { backup.close(); }
    const created = store.createSessions([{ title: '新版独立测试', source: 'manual', messages: session.messages }])[0];
    const analysis = store.createAnalyses([created.id])[0];
    store.markRunning(analysis.id);
    const result = fixture(); result.post_id = created.id;
    const actual = deploymentIdentity(v2Deployment);
    store.succeed(analysis.id, result, actual, 12);
    const saved = store.detail(analysis.id);
    assert.equal(saved.analysis.modelVersion, actual.modelVersion);
    assert.equal(saved.analysis.modelHash, actual.modelHash);
    assert.equal(saved.analysis.deploymentHash, actual.deploymentHash);
    assert.equal(saved.analysis.schemaVersion, 2);
    assert.equal(saved.analysis.riskScore, 0.8);
    assert.equal(saved.analysis.threshold, null);
    assert.equal(saved.analysis.scoreType, 'max_message_category_score');
  } finally { store.close(); await rm(dir, { recursive: true, force: true }); }
});

test('schema2 HTTP storage and exports preserve category scores and actual worker identity', async () => {
  const dir = await mkdtemp(path.join(tmpdir(), 'campus-v2-http-'));
  const actual = deploymentIdentity(v2Deployment);
  const instance = createApp({ dataDir: dir, engine: {
    health: () => ({ status: 'ready', ...actual }), start: async () => {}, close: async () => {},
    analyze: async current => { const result = fixture(); result.post_id = current.id; for (const message of [...result.message_scores, ...result.evidence, ...result.riskCategories.flatMap(category => category.evidence)]) message.post_id = current.id; return result; },
  } });
  const server = instance.app.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const address = server.address(); assert.ok(address && typeof address !== 'string');
  const base = `http://127.0.0.1:${address.port}`;
  const post = async (route: string, body: unknown) => (await fetch(base + route, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) })).json();
  try {
    const created = await post('/api/sessions', { sessions: [{ title: '独立v2协议测试', source: 'manual', messages: session.messages }] });
    const queued = await post('/api/analyses', { sessionIds: [created.sessions[0].id] });
    const id = queued.analyses[0].id;
    let detail;
    for (let index = 0; index < 50; index++) { detail = await (await fetch(base + `/api/analyses/${id}`)).json(); if (detail.analysis.status === 'succeeded') break; await new Promise(resolve => setTimeout(resolve, 10)); }
    assert.equal(detail.analysis.status, 'succeeded');
    assert.equal(detail.analysis.modelVersion, 'test-v2');
    assert.equal(detail.analysis.deploymentHash, actual.deploymentHash);
    assert.equal(detail.result.overallEvidenceScore, 0.8);
    assert.equal(detail.result.conversationRisk.score, 0.3);
    assert.ok(!('risk_probability' in detail.result));
    const csv = await (await fetch(base + `/api/analyses/${id}/export?format=csv`)).text();
    assert.ok(csv.includes('max_message_category_score'));
    assert.ok(csv.includes('conversation_reference_score'));
    assert.ok(csv.includes('message_category_scores'));
  } finally { await new Promise<void>(resolve => server.close(() => resolve())); await instance.close(); await rm(dir, { recursive: true, force: true }); }
});

test('encoder warmup uses one empty-string public encode call and never invokes a classifier', async () => {
  const python = process.env.CST_PYTHON ?? path.resolve('../..', 'cst_mil/.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
  const program = `import importlib.util, pathlib, sys
spec=importlib.util.spec_from_file_location('campus_worker', pathlib.Path('python/worker.py'))
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
class Encoder:
    def __init__(self): self.calls=[]
    def encode(self, texts): self.calls.append(texts); return [[0.0]]
class Model:
    def __init__(self): self.instance=Encoder(); self.encoder_calls=0
    def encoder(self): self.encoder_calls+=1; return self.instance
    def predict(self,*args): raise AssertionError('classifier must not run')
model=Model(); module.warm_encoder(model)
assert model.encoder_calls==1 and model.instance.calls==[['']]
print('warmup-public-api-only')`;
  const result = await promisify(execFile)(python, ['-B', '-X', 'utf8', '-c', program], { windowsHide: true });
  assert.match(result.stdout, /warmup-public-api-only/);
});

test('startup to ready and first request wall times are recorded separately', async () => {
  const actual = deploymentIdentity(v2Deployment);
  const child = Object.assign(new EventEmitter(), { stdout: new PassThrough(), stderr: new PassThrough(), stdin: new PassThrough(), kill: () => true });
  child.stdin.on('data', chunk => {
    const request = JSON.parse(chunk.toString());
    if (request.type === 'analyze') setTimeout(() => child.stdout.write(JSON.stringify({ type: 'result', ...actual, id: request.id, result: fixture() }) + '\n'), 10);
  });
  const worker = new PythonWorker({ deployment: v2Deployment, spawnProcess: () => {
    setTimeout(() => child.stdout.write(JSON.stringify({ type: 'ready', ...actual, thresholds: v2Deployment.thresholds, conversationThreshold: v2Deployment.conversationThreshold, encoderWarmed: true, startupSeconds: 0.02 }) + '\n'), 20);
    return child as unknown as ChildProcessWithoutNullStreams;
  } });
  try {
    await worker.start();
    assert.ok((worker.health().startupDurationMs ?? 0) >= 10);
    assert.equal(worker.health().pythonStartupSeconds, 0.02);
    assert.equal(worker.health().encoderWarmed, true);
    assert.equal(worker.health().firstRequestDurationMs, undefined);
    await worker.analyze(session);
    const first = worker.health().firstRequestDurationMs;
    assert.ok((first ?? 0) >= 5); assert.equal(worker.health().firstRequestStatus, 'succeeded');
    await worker.analyze(session); assert.equal(worker.health().firstRequestDurationMs, first);
  } finally { const closing = worker.close(); child.emit('close', 0); await closing; }
});
