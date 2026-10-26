import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { defaultDeploymentPath, loadDeployment, REPO_ROOT, resolveRepoPath } from '../config/deployment.mjs';

test('public asset catalog fixes model, encoder revision and research source fingerprints', () => {
  const catalog = JSON.parse(readFileSync(path.join(REPO_ROOT, 'cst_mil/src/cst_mil/deployment/catalog.json'), 'utf8'));
  assert.equal(catalog.release, 'demo-2026.10.15');
  assert.equal(catalog.model.sha256, 'f93a8dd1698c983d649f270f1c5e19780939b91e3dc10aed4a1aa1a7e595bb32');
  assert.equal(catalog.encoder.revision, 'e8f8c211226b894fcb81acc59f3b34ba3efd5f42');
  assert.ok(Object.keys(catalog.codeFingerprints).length >= 60);
  for (const [relative, hash] of Object.entries(catalog.codeFingerprints)) {
    assert.equal(createHash('sha256').update(readFileSync(resolveRepoPath(relative))).digest('hex'), hash, relative);
  }
});

test('explicit deployment selection is preserved and unauthorized interfaces fail before any asset loading', () => {
  const before = process.env.CAMPUS_DEPLOYMENT_FILE;
  const directory = mkdtempSync(path.join(tmpdir(), 'cst-public-contract-'));
  const file = path.join(directory, 'bad.json');
  try {
    process.env.CAMPUS_DEPLOYMENT_FILE = file;
    assert.equal(defaultDeploymentPath(), file);
    for (const predictorModule of ['untrusted.module', 'cst_mil.deployment.not_adapter']) {
      writeFileSync(file, JSON.stringify({ manifestVersion: 1, schemaVersion: 2, predictorModule,
        modelClass: 'FollowupPackage', predictFunction: 'predict_session' }));
      assert.throws(() => loadDeployment(file), /interface or model type/);
    }
  } finally {
    if (before === undefined) delete process.env.CAMPUS_DEPLOYMENT_FILE;
    else process.env.CAMPUS_DEPLOYMENT_FILE = before;
    rmSync(directory, { recursive: true, force: true });
  }
});

test('deployment repository paths cannot escape through an absolute path or parent traversal', () => {
  assert.throws(() => resolveRepoPath('../private'), /stay inside/);
  assert.throws(() => resolveRepoPath(path.resolve('outside')), /relative/);
});
