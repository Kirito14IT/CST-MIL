import { createHash } from 'node:crypto';
import { existsSync, readFileSync, realpathSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const APP_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const REPO_ROOT = path.resolve(APP_ROOT, '..');
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const hashPattern = /^[a-f0-9]{64}$/;
const allowed = { 1: [['cst_mil.bench.model', 'BenchmarkModel']], 2: [['cst_mil.unified_semantic_v1_1.model', 'UnifiedSemanticModel'], ['cst_mil.demo_semantic_adapter', 'FollowupPackage'], ['cst_mil.deployment.adapter', 'FollowupPackage']] };

export function defaultDeploymentPath() {
  if (process.env.CAMPUS_DEPLOYMENT_FILE) return process.env.CAMPUS_DEPLOYMENT_FILE;
  const portable = path.join(REPO_ROOT, '.runtime', 'deployment.json');
  if (existsSync(portable)) return portable;
  const original = path.join(APP_ROOT, 'config', 'deployment.json');
  if (existsSync(original)) {
    const originalManifest = JSON.parse(readFileSync(original, 'utf8').replace(/^\uFEFF/, ''));
    if (typeof originalManifest.modelPath === 'string' && existsSync(path.resolve(REPO_ROOT, originalManifest.modelPath))) return original;
  }
  throw new Error('Model assets are missing. Run python -m cst_mil.deployment setup from the repository root, then npm start.');
}

export function resolveRepoPath(relative) {
  if (typeof relative !== 'string' || !relative || path.isAbsolute(relative) || relative.includes('\0')) throw new Error('Deployment paths must be relative to the repository root');
  const absolute = path.resolve(REPO_ROOT, relative);
  const within = value => { const rel = path.relative(REPO_ROOT, value); return rel !== '..' && !rel.startsWith('..' + path.sep) && !path.isAbsolute(rel); };
  if (!within(absolute)) throw new Error('Deployment paths must stay inside the repository');
  const real = realpathSync(absolute);
  if (!within(real)) throw new Error('Deployment assets must not escape the repository through links');
  return real;
}

export function loadDeployment(file = defaultDeploymentPath()) {
  const bytes = readFileSync(file);
  const manifest = JSON.parse(bytes.toString('utf8').replace(/^\uFEFF/, ''));
  const schema = allowed[manifest.schemaVersion];
  if (manifest.manifestVersion !== 1 || !schema?.some(([module, modelClass]) => manifest.predictorModule === module && manifest.modelClass === modelClass) || manifest.predictFunction !== 'predict_session') throw new Error('Unsupported deployment interface or model type');
  if (![manifest.modelVersion, manifest.taskName].every(value => typeof value === 'string' && value.trim())) throw new Error('Deployment manifest is missing the model version or task name');
  if (!hashPattern.test(manifest.modelSha256)) throw new Error('Invalid model fingerprint in deployment manifest');
  const modelAbsolutePath = resolveRepoPath(manifest.modelPath);
  if (sha256(readFileSync(modelAbsolutePath)) !== manifest.modelSha256) throw new Error('Model fingerprint does not match the deployment manifest');
  if (!manifest.codeFingerprints || typeof manifest.codeFingerprints !== 'object' || Array.isArray(manifest.codeFingerprints) || !Object.keys(manifest.codeFingerprints).length) throw new Error('Deployment manifest is missing source fingerprints');
  for (const [relative, expected] of Object.entries(manifest.codeFingerprints)) {
    if (typeof expected !== 'string' || !hashPattern.test(expected) || sha256(readFileSync(resolveRepoPath(relative))) !== expected) throw new Error(`Deployment source fingerprint mismatch: ${relative}`);
  }
  if (manifest.assetFingerprints !== undefined) {
    if (!manifest.assetFingerprints || typeof manifest.assetFingerprints !== 'object' || Array.isArray(manifest.assetFingerprints)) throw new Error('Invalid deployment asset fingerprint format');
    for (const [relative, expected] of Object.entries(manifest.assetFingerprints)) {
      if (typeof expected !== 'string' || !hashPattern.test(expected) || sha256(readFileSync(resolveRepoPath(relative))) !== expected) throw new Error(`Deployment asset fingerprint mismatch: ${relative}`);
    }
  }
  if (manifest.modelClass === 'FollowupPackage') {
    if (!manifest.bundle || manifest.bundle.seed !== 42 || manifest.bundle.architecture !== 'full' || manifest.modelVersion !== 'semantic-patent-followup-v1' || ![manifest.bundle.manifestPath, manifest.bundle.explanationPath, manifest.encoder?.manifestPath].every(relative => typeof relative === 'string' && hashPattern.test(manifest.assetFingerprints?.[relative] ?? ''))) throw new Error('Semantic deployment is missing the frozen model bundle or explanation binding');
    const bundle = JSON.parse(readFileSync(resolveRepoPath(manifest.bundle.manifestPath), 'utf8').replace(/^\uFEFF/, ''));
    const explanation = JSON.parse(readFileSync(resolveRepoPath(manifest.bundle.explanationPath), 'utf8').replace(/^\uFEFF/, ''));
    if (bundle.model_sha256 !== manifest.modelSha256 || bundle.seed !== manifest.bundle.seed || bundle.architecture !== manifest.bundle.architecture || JSON.stringify(explanation) !== JSON.stringify(bundle.explanation_config)) throw new Error('Semantic model bundle identity or explanation configuration mismatch');
    const encoderInfo = JSON.parse(readFileSync(resolveRepoPath(manifest.encoder.manifestPath), 'utf8').replace(/^\uFEFF/, ''));
    if (encoderInfo.model_fingerprint !== manifest.encoder.fingerprint || bundle.encoder.fingerprint !== manifest.encoder.fingerprint) throw new Error('Semantic encoder manifest identity mismatch');
    const expectedThresholds = Object.fromEntries(['attack', 'hate', 'spam_scam'].map(key => [key, bundle.thresholds[key]]));
    if (Object.keys(expectedThresholds).some(key => manifest.thresholds?.[key] !== expectedThresholds[key]) || manifest.conversationThreshold !== bundle.thresholds.conversation || JSON.stringify(manifest.trainingSources) !== JSON.stringify(bundle.training_sources)) throw new Error('Semantic thresholds or training sources differ from the model bundle');
    if (manifest.predictorModule === 'cst_mil.deployment.adapter') {
      const catalogBytes = readFileSync(resolveRepoPath('cst_mil/src/cst_mil/deployment/catalog.json'));
      const catalog = JSON.parse(catalogBytes.toString('utf8'));
      const portable = manifest.portable;
      if (!portable || catalog.catalogVersion !== 1 || catalog.release !== 'demo-2026.10.15' || portable.release !== catalog.release || portable.catalogSha256 !== sha256(catalogBytes) || portable.archiveSha256 !== catalog.archive.sha256 || manifest.modelSha256 !== catalog.model.sha256 || manifest.encoder.fingerprint !== catalog.encoder.fingerprint || JSON.stringify(manifest.codeFingerprints) !== JSON.stringify(catalog.codeFingerprints)) throw new Error('Portable deployment differs from the public release catalog');
      const published = readFileSync(resolveRepoPath(portable.publishedManifestPath));
      if (sha256(published) !== catalog.archive_files['model_manifest.json'].sha256 || manifest.assetFingerprints[portable.publishedManifestPath] !== sha256(published)) throw new Error('Published technical manifest fingerprint mismatch');
      const technical = JSON.parse(published.toString('utf8'));
      const normalized = structuredClone(bundle);
      normalized.encoder.encoder_path = technical.encoder.encoder_path;
      normalized.encoder.asset_hashes = technical.encoder.asset_hashes;
      if (JSON.stringify(normalized) !== JSON.stringify(technical) || JSON.stringify(encoderInfo) !== JSON.stringify(catalog.encoder.manifest)) throw new Error('Portable runtime may only rebind verified encoder paths');
    } else if (manifest.portable !== undefined) throw new Error('Portable binding requires the dedicated prediction adapter');
  }
  const thresholdKeys = manifest.schemaVersion === 2 ? ['attack', 'hate', 'spam_scam'] : ['attack'];
  if (!manifest.thresholds || typeof manifest.thresholds !== 'object' || Array.isArray(manifest.thresholds) || Object.keys(manifest.thresholds).sort().join(',') !== thresholdKeys.sort().join(',') || !Object.values(manifest.thresholds).every(value => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1)) throw new Error('Invalid deployment category thresholds');
  if (typeof manifest.conversationThreshold !== 'number' || !Number.isFinite(manifest.conversationThreshold) || manifest.conversationThreshold < 0 || manifest.conversationThreshold > 1) throw new Error('Invalid deployment conversation threshold');
  if (!Array.isArray(manifest.trainingSources) || !manifest.trainingSources.length || manifest.trainingSources.some(value => typeof value !== 'string')) throw new Error('Invalid deployment training sources');
  let encoderAbsolutePath = null;
  if (manifest.schemaVersion === 2) {
    if (!manifest.encoder || !hashPattern.test(manifest.encoder.fingerprint)) throw new Error('Semantic model is missing encoder asset identity');
    encoderAbsolutePath = resolveRepoPath(manifest.encoder.path);
    if (manifest.encoder.manifestPath) resolveRepoPath(manifest.encoder.manifestPath);
  } else if (manifest.encoder !== null) throw new Error('Legacy deployment must not specify an encoder');
  return { ...manifest, deploymentHash: sha256(bytes), manifestPath: path.resolve(file), modelAbsolutePath, encoderAbsolutePath };
}

export function deploymentIdentity(deployment) {
  return { modelVersion: deployment.modelVersion, modelHash: deployment.modelSha256, deploymentHash: deployment.deploymentHash, schemaVersion: deployment.schemaVersion, taskName: deployment.taskName };
}

export function matchesDeployment(value, deployment) {
  const identity = deploymentIdentity(deployment);
  return !!value && Object.entries(identity).every(([key, expected]) => value[key] === expected);
}

export function assertReusableService(health, deployment) {
  if (health?.app !== 'Campus Text Lab' || health.appProtocolVersion !== 2 || health.worker?.status !== 'ready' || !matchesDeployment(health.worker, deployment)) throw new Error('This port is occupied by an unverified or older service. Stop it with Ctrl+C in its original terminal, then run npm start. Other processes will not be stopped automatically.');
}
