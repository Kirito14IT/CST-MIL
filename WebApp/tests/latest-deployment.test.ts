import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { loadDeployment } from '../config/deployment.mjs';
import { validatePrediction } from '../server/prediction.ts';
import type { PredictionV2, SessionRecord } from '../shared/types.ts';

test('full-data deployment explicitly binds FollowupPackage, seed42 and frozen explanation assets', async () => {
  const manifest = loadDeployment(path.resolve('config/deployment.latest-full.json'));
  assert.equal(manifest.modelClass, 'FollowupPackage');
  assert.equal(manifest.predictorModule, 'cst_mil.demo_semantic_adapter');
  assert.equal(manifest.modelVersion, 'semantic-patent-followup-v1');
  assert.equal(manifest.modelSha256, '39e842dbffe10356c6dbef4f17bb320868ef7b007c7b1972a38c315d4af87a59');
  assert.equal(manifest.bundle?.seed, 42);
  assert.equal(manifest.trainingSources.length, 5);
  const legacy = loadDeployment(path.resolve('config/deployment.legacy-r0.json'));
  assert.equal(legacy.schemaVersion, 1);
  const dir = await mkdtemp(path.join(tmpdir(), 'campus-latest-manifest-'));
  try {
    for (const changes of [
      { modelClass: 'UnifiedSemanticModel' },
      { modelVersion: 'cst-unified-semantic-v1.1' },
      { bundle: { ...manifest.bundle, seed: 43 } },
      { assetFingerprints: {} },
      { thresholds: { ...manifest.thresholds, attack: 0.5 } },
      { trainingSources: ['sccd'] },
    ]) {
      const file = path.join(dir, 'deployment.json');
      await writeFile(file, JSON.stringify({ ...manifest, ...changes }));
      assert.throws(() => loadDeployment(file));
    }
    const defaultManifest = JSON.parse(await readFile('config/deployment.json', 'utf8'));
    assert.ok([1, 2].includes(defaultManifest.schemaVersion));
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('latest adapter preserves one-pass semantic encoding, label isolation and traceable reply associations', { timeout: 30000 }, async () => {
  const python = process.env.CST_PYTHON ?? path.resolve('..', 'cst_mil/.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
  const result = await promisify(execFile)(python, ['-B', '-X', 'utf8', 'python/test_latest_adapter.py'], { windowsHide: true, env: { ...process.env, OMP_NUM_THREADS: '4', MKL_NUM_THREADS: '4', OPENBLAS_NUM_THREADS: '4' } });
  assert.match(result.stderr, /Ran 7 tests/);
  assert.match(result.stderr, /OK/);
});

test('schema2 full-data contract rejects forged association membership, copied parent strength and changed input reply edges', () => {
  const manifest = loadDeployment(path.resolve('config/deployment.latest-full.json'));
  const input: SessionRecord = { id: 's', title: '示例', source: 'manual', createdAt: '2026-10-15', messageCount: 3,
    messages: [{ id: 'p', text: '东北锅包肉好吃' }, { id: 'b', text: '东北锅包肉很好吃' }, { id: 'c', text: '是的', parentId: 'p' }] };
  const scores = [.2, .3, .15];
  const messages = input.messages.map((message, index) => ({ post_id: 's', comment_id: message.id, raw_text: message.text,
    parent_id: message.parentId ?? null, order: index, position: index, comment_time: null, speaker_id: null,
    risk_score: scores[index], riskCategoryScores: { attack: scores[index], hate: .1, spam_scam: .1 },
    topic_ids: [0], topic_strengths: [index === 2 ? 0 : 1], assigned: true, rank: [2, 1, 3][index],
    assignment_provenance: index === 2 ? 'direct_parent_context' as const : 'semantic_cluster' as const,
    context_parent_message_id: index === 2 ? 'p' : null }));
  const valid: PredictionV2 = { post_id: 's', split: 'inference', label: null, schemaVersion: 2, modelVersion: manifest.modelVersion,
    prediction: 0, prediction_name: '未发现明确风险线索', overallEvidenceScore: .3, scoreType: 'max_message_category_score',
    topics: [{ topic_id: 0, title: '东北', keywords: ['东北'], topic_strength: 1, risk_score: .65 / 3,
      message_ids: ['p', 'b', 'c'], basic_message_ids: ['p', 'b'], context_association_message_ids: ['c'],
      support_messages: messages, evidence: messages, segments: [], keyword_supports: { 东北: [
        { message_id: 'p', start: 0, end: 2, excerpt: '东北' }, { message_id: 'b', start: 0, end: 2, excerpt: '东北' },
      ] } }], message_scores: messages, evidence: [messages[1], messages[0], messages[2]],
    riskCategories: (['attack', 'hate', 'spam_scam'] as const).map(id => ({ id, name: id, score: id === 'attack' ? .3 : .1,
      threshold: manifest.thresholds[id], detected: false, message_ids: [], evidence: [] })),
    conversationRisk: { score: .3, threshold: manifest.conversationThreshold, prediction: 0, domain: 'SCCD', note: '迁移参考' },
    coverage: 1, coverage_details: { assigned_messages: 3, total_messages: 3, fraction: 1, unassigned_message_ids: [], zero_topic_messages: 0 } };
  assert.doesNotThrow(() => validatePrediction(valid, manifest, 's', input));
  const changes: ((result: PredictionV2) => void)[] = [
    r => { r.topics[0].basic_message_ids!.push('c'); },
    r => { r.message_scores[2].context_parent_message_id = 'b'; },
    r => { r.message_scores[2].topic_strengths[0] = .9; },
    r => { r.topics[0].keyword_supports!.东北[0].message_id = 'c'; },
    r => { r.message_scores[2].parent_id = 'b'; },
    r => { r.message_scores[2].topic_ids = []; r.message_scores[2].assigned = false; },
    r => { delete r.topics[0].basic_message_ids; },
  ];
  for (const change of changes) { const bad = structuredClone(valid); change(bad); assert.throws(() => validatePrediction(bad, manifest, 's', input)); }
});
