import { createHash } from 'node:crypto';
import { existsSync, readFileSync, realpathSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const APP_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const REPO_ROOT = path.resolve(APP_ROOT, '..', '..');
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
  throw new Error('模型资产尚未安装。请先在仓库根目录执行 python -m cst_mil.deployment setup，再运行 npm start。');
}

export function resolveRepoPath(relative) {
  if (typeof relative !== 'string' || !relative || path.isAbsolute(relative) || relative.includes('\0')) throw new Error('部署路径必须相对仓库根目录');
  const absolute = path.resolve(REPO_ROOT, relative);
  const within = value => { const rel = path.relative(REPO_ROOT, value); return rel !== '..' && !rel.startsWith('..' + path.sep) && !path.isAbsolute(rel); };
  if (!within(absolute)) throw new Error('部署路径不能越出仓库');
  const real = realpathSync(absolute);
  if (!within(real)) throw new Error('部署资产不能通过链接越出仓库');
  return real;
}

export function loadDeployment(file = defaultDeploymentPath()) {
  const bytes = readFileSync(file);
  const manifest = JSON.parse(bytes.toString('utf8').replace(/^\uFEFF/, ''));
  const schema = allowed[manifest.schemaVersion];
  if (manifest.manifestVersion !== 1 || !schema?.some(([module, modelClass]) => manifest.predictorModule === module && manifest.modelClass === modelClass) || manifest.predictFunction !== 'predict_session') throw new Error('部署清单的接口或模型类型不受支持');
  if (![manifest.modelVersion, manifest.taskName].every(value => typeof value === 'string' && value.trim())) throw new Error('部署清单缺少模型版本或任务名');
  if (!hashPattern.test(manifest.modelSha256)) throw new Error('部署清单模型指纹无效');
  const modelAbsolutePath = resolveRepoPath(manifest.modelPath);
  if (sha256(readFileSync(modelAbsolutePath)) !== manifest.modelSha256) throw new Error('模型文件与部署清单指纹不一致');
  if (!manifest.codeFingerprints || typeof manifest.codeFingerprints !== 'object' || Array.isArray(manifest.codeFingerprints) || !Object.keys(manifest.codeFingerprints).length) throw new Error('部署清单缺少源码指纹');
  for (const [relative, expected] of Object.entries(manifest.codeFingerprints)) {
    if (typeof expected !== 'string' || !hashPattern.test(expected) || sha256(readFileSync(resolveRepoPath(relative))) !== expected) throw new Error(`部署源码指纹不一致：${relative}`);
  }
  if (manifest.assetFingerprints !== undefined) {
    if (!manifest.assetFingerprints || typeof manifest.assetFingerprints !== 'object' || Array.isArray(manifest.assetFingerprints)) throw new Error('部署资产指纹格式无效');
    for (const [relative, expected] of Object.entries(manifest.assetFingerprints)) {
      if (typeof expected !== 'string' || !hashPattern.test(expected) || sha256(readFileSync(resolveRepoPath(relative))) !== expected) throw new Error(`部署资产指纹不一致：${relative}`);
    }
  }
  if (manifest.modelClass === 'FollowupPackage') {
    if (!manifest.bundle || manifest.bundle.seed !== 42 || manifest.bundle.architecture !== 'full' || manifest.modelVersion !== 'semantic-patent-followup-v1' || ![manifest.bundle.manifestPath, manifest.bundle.explanationPath, manifest.encoder?.manifestPath].every(relative => typeof relative === 'string' && hashPattern.test(manifest.assetFingerprints?.[relative] ?? ''))) throw new Error('全量语义部署缺少冻结研究包及解释绑定');
    const bundle = JSON.parse(readFileSync(resolveRepoPath(manifest.bundle.manifestPath), 'utf8').replace(/^\uFEFF/, ''));
    const explanation = JSON.parse(readFileSync(resolveRepoPath(manifest.bundle.explanationPath), 'utf8').replace(/^\uFEFF/, ''));
    if (bundle.model_sha256 !== manifest.modelSha256 || bundle.seed !== manifest.bundle.seed || bundle.architecture !== manifest.bundle.architecture || JSON.stringify(explanation) !== JSON.stringify(bundle.explanation_config)) throw new Error('全量语义研究包身份或解释配置不一致');
    const encoderInfo = JSON.parse(readFileSync(resolveRepoPath(manifest.encoder.manifestPath), 'utf8').replace(/^\uFEFF/, ''));
    if (encoderInfo.model_fingerprint !== manifest.encoder.fingerprint || bundle.encoder.fingerprint !== manifest.encoder.fingerprint) throw new Error('全量语义编码器清单身份不一致');
    const expectedThresholds = Object.fromEntries(['attack', 'hate', 'spam_scam'].map(key => [key, bundle.thresholds[key]]));
    if (Object.keys(expectedThresholds).some(key => manifest.thresholds?.[key] !== expectedThresholds[key]) || manifest.conversationThreshold !== bundle.thresholds.conversation || JSON.stringify(manifest.trainingSources) !== JSON.stringify(bundle.training_sources)) throw new Error('全量语义阈值或训练来源与研究包不一致');
    if (manifest.predictorModule === 'cst_mil.deployment.adapter') {
      const catalogBytes = readFileSync(resolveRepoPath('cst_mil/src/cst_mil/deployment/catalog.json'));
      const catalog = JSON.parse(catalogBytes.toString('utf8'));
      const portable = manifest.portable;
      if (!portable || catalog.catalogVersion !== 1 || catalog.release !== 'demo-2026.10.15' || portable.release !== catalog.release || portable.catalogSha256 !== sha256(catalogBytes) || portable.archiveSha256 !== catalog.archive.sha256 || manifest.modelSha256 !== catalog.model.sha256 || manifest.encoder.fingerprint !== catalog.encoder.fingerprint || JSON.stringify(manifest.codeFingerprints) !== JSON.stringify(catalog.codeFingerprints)) throw new Error('便携部署与公开发布清单不一致');
      const published = readFileSync(resolveRepoPath(portable.publishedManifestPath));
      if (sha256(published) !== catalog.archive_files['model_manifest.json'].sha256 || manifest.assetFingerprints[portable.publishedManifestPath] !== sha256(published)) throw new Error('原始发布技术清单指纹不匹配');
      const technical = JSON.parse(published.toString('utf8'));
      const normalized = structuredClone(bundle);
      normalized.encoder.encoder_path = technical.encoder.encoder_path;
      normalized.encoder.asset_hashes = technical.encoder.asset_hashes;
      if (JSON.stringify(normalized) !== JSON.stringify(technical) || JSON.stringify(encoderInfo) !== JSON.stringify(catalog.encoder.manifest)) throw new Error('便携运行时只允许重绑定已核验的编码器路径');
    } else if (manifest.portable !== undefined) throw new Error('便携绑定必须使用专用预测适配器');
  }
  const thresholdKeys = manifest.schemaVersion === 2 ? ['attack', 'hate', 'spam_scam'] : ['attack'];
  if (!manifest.thresholds || typeof manifest.thresholds !== 'object' || Array.isArray(manifest.thresholds) || Object.keys(manifest.thresholds).sort().join(',') !== thresholdKeys.sort().join(',') || !Object.values(manifest.thresholds).every(value => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1)) throw new Error('部署类别阈值无效');
  if (typeof manifest.conversationThreshold !== 'number' || !Number.isFinite(manifest.conversationThreshold) || manifest.conversationThreshold < 0 || manifest.conversationThreshold > 1) throw new Error('部署会话阈值无效');
  if (!Array.isArray(manifest.trainingSources) || !manifest.trainingSources.length || manifest.trainingSources.some(value => typeof value !== 'string')) throw new Error('部署训练来源无效');
  let encoderAbsolutePath = null;
  if (manifest.schemaVersion === 2) {
    if (!manifest.encoder || !hashPattern.test(manifest.encoder.fingerprint)) throw new Error('新模型缺少编码器资产身份');
    encoderAbsolutePath = resolveRepoPath(manifest.encoder.path);
    if (manifest.encoder.manifestPath) resolveRepoPath(manifest.encoder.manifestPath);
  } else if (manifest.encoder !== null) throw new Error('旧模型部署编码器应为空');
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
  if (health?.app !== 'Campus Text Lab' || health.appProtocolVersion !== 2 || health.worker?.status !== 'ready' || !matchesDeployment(health.worker, deployment)) throw new Error('此端口已有旧模型或未核验的校园服务。请在其原启动窗口按 Ctrl+C 关闭，再执行 npm start；不会自动关闭其他进程。');
}
