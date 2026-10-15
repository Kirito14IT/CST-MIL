import path from 'node:path';
import { loadDeployment, APP_ROOT, REPO_ROOT } from '../config/deployment.mjs';
export { APP_ROOT, REPO_ROOT } from '../config/deployment.mjs';

export const RESEARCH_ROOT = path.join(REPO_ROOT, 'cst_mil');
export const FOLLOWUP_ROOT = path.join(RESEARCH_ROOT, 'benchmark', 'research_followup', '20260922');
export const DEPLOYMENT = loadDeployment();
export const MODEL_PATH = DEPLOYMENT.modelAbsolutePath;
export const MODEL_HASH = DEPLOYMENT.modelSha256;
export const MODEL_VERSION = DEPLOYMENT.modelVersion;
