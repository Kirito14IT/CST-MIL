import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { createInterface } from 'node:readline';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import type { Prediction, SessionRecord, WorkerHealth } from '../shared/types.ts';
import { APP_ROOT, DEPLOYMENT, RESEARCH_ROOT } from './paths.ts';
import { matchesDeployment, type LoadedDeployment } from '../config/deployment.mjs';
import { validatePrediction } from './prediction.ts';
export { MODEL_HASH } from './paths.ts';

export interface InferenceEngine {
  health(): WorkerHealth;
  start(): Promise<void>;
  analyze(session: SessionRecord): Promise<Prediction>;
  close(): Promise<void>;
}
interface Options { pythonPath?: string; startupTimeoutMs?: number; requestTimeoutMs?: number; deployment?: LoadedDeployment; spawnProcess?: (command: string, args: string[]) => ChildProcessWithoutNullStreams }
interface Pending { id: string; session: SessionRecord; startedAt: number; resolve: (value: Prediction) => void; reject: (error: Error) => void; timer: NodeJS.Timeout }
export class PythonWorker implements InferenceEngine {
  private child: ChildProcessWithoutNullStreams | null = null;
  private state: WorkerHealth = { status: 'starting' };
  private closed = false;
  private starting: Promise<void> | null = null;
  private pending: Pending | null = null;
  private failStartup: ((error: Error) => void) | null = null;
  private options: Options;
  private deployment: LoadedDeployment;
  constructor(options: Options = {}) { this.options = options; this.deployment = options.deployment ?? DEPLOYMENT; }
  health(): WorkerHealth { return { ...this.state }; }
  async start(): Promise<void> {
    if (this.closed) throw new Error('The inference service is closed');
    if (this.child && this.state.status === 'ready') return;
    if (this.starting) return this.starting;
    this.state = { status: 'starting' };
    this.starting = this.launch();
    try { await this.starting; } finally { this.starting = null; }
  }
  private launch(): Promise<void> {
    const startupStarted = performance.now();
    return new Promise((resolve, reject) => {
      let settled = false;
      let startupTimer: NodeJS.Timeout;
      const fail = (error: Error) => {
        if (this.child !== child) return;
        this.state = { ...this.state, status: 'error', error: error.message };
        if (!settled) { settled = true; clearTimeout(startupTimer); reject(error); }
        if (this.pending) { if (this.state.firstRequestDurationMs === undefined) { this.state.firstRequestDurationMs = performance.now() - this.pending.startedAt; this.state.firstRequestStatus = 'failed'; } clearTimeout(this.pending.timer); this.pending.reject(error); this.pending = null; }
      };
      this.failStartup = fail;
      const python = this.options.pythonPath ?? process.env.CST_PYTHON ?? path.join(RESEARCH_ROOT, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
      const args = ['-u', '-B', '-X', 'utf8', path.join(APP_ROOT, 'python', 'worker.py'), this.deployment.manifestPath];
      const child = this.options.spawnProcess ? this.options.spawnProcess(python, args) : spawn(python, args, {
        cwd: APP_ROOT, shell: false, windowsHide: true,
        env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1' }, stdio: ['pipe', 'pipe', 'pipe'],
      });
      this.child = child;
      startupTimer = setTimeout(() => { fail(new Error('Model initialization timed out. Check the Python environment and model files')); child.kill(); }, this.options.startupTimeoutMs ?? 90000);
      const reader = createInterface({ input: child.stdout, crlfDelay: Infinity });
      reader.on('line', line => {
        if (this.child !== child) return;
        let message: Record<string, unknown>;
        try { message = JSON.parse(line) as Record<string, unknown>; }
        catch { fail(new Error('The inference worker returned non-JSON data')); child.kill(); return; }
        if (message.type === 'ready') {
          const thresholds = message.thresholds as Record<string, unknown> | undefined;
          if (!matchesDeployment(message, this.deployment) || !thresholds || Object.keys(thresholds).sort().join(',') !== Object.keys(this.deployment.thresholds).sort().join(',') || Object.entries(this.deployment.thresholds).some(([id, threshold]) => thresholds[id] !== threshold) || message.conversationThreshold !== this.deployment.conversationThreshold || (this.deployment.schemaVersion === 2 && message.encoderWarmed !== true)) { fail(new Error('Ready-state identity, thresholds or encoder warmup do not match the deployment manifest')); child.kill(); return; }
          if (settled) return;
          settled = true;
          clearTimeout(startupTimer);
          this.state = { status: 'ready', modelHash: String(message.modelHash), modelVersion: String(message.modelVersion), deploymentHash: String(message.deploymentHash), schemaVersion: this.deployment.schemaVersion, taskName: String(message.taskName), thresholds: { ...this.deployment.thresholds }, conversationThreshold: this.deployment.conversationThreshold, startupDurationMs: performance.now() - startupStarted, ...(typeof message.startupSeconds === 'number' && Number.isFinite(message.startupSeconds) ? { pythonStartupSeconds: message.startupSeconds } : {}), encoderWarmed: message.encoderWarmed === true, ...(this.deployment.schemaVersion === 1 ? { threshold: this.deployment.thresholds.attack } : {}) };
          resolve();
        } else if (message.type === 'fatal') {
          fail(new Error(typeof message.error === 'string' ? message.error : 'Model startup failed'));
          child.kill();
        } else if (this.pending && message.id === this.pending.id) {
          const pending = this.pending;
          this.pending = null;
          clearTimeout(pending.timer);
          const firstRequest = this.state.firstRequestDurationMs === undefined;
          if (firstRequest) { this.state.firstRequestDurationMs = performance.now() - pending.startedAt; this.state.firstRequestStatus = 'failed'; }
          if (message.type === 'error') pending.reject(new Error(typeof message.error === 'string' ? message.error : 'Model inference failed'));
          else if (message.type === 'result' && message.result && typeof message.result === 'object') {
            try {
              if (!matchesDeployment(message, this.deployment)) throw new Error('Prediction identity does not match the loaded model');
              const prediction = validatePrediction(message.result, this.deployment, pending.session.id, pending.session);
              if (firstRequest) { this.state.firstRequestDurationMs = performance.now() - pending.startedAt; this.state.firstRequestStatus = 'succeeded'; }
              pending.resolve(prediction);
            } catch (error) { pending.reject(error instanceof Error ? error : new Error('Model output schema validation failed')); }
          } else pending.reject(new Error('Invalid model response type'));
        }
      });
      let stderr = '';
      child.stderr.on('data', (buffer: Buffer) => { stderr = (stderr + buffer.toString('utf8')).slice(-8000); });
      child.on('error', error => {
        if (this.child !== child) return;
        fail(new Error(`Could not start the Python inference worker: ${error.message}`));
        if (this.child === child) this.child = null;
      });
      child.stdin.on('error', () => {
        if (this.child !== child) return;
        fail(new Error('The Python communication pipe is closed'));
        child.kill();
      });
      child.on('close', code => {
        reader.close();
        clearTimeout(startupTimer);
        if (this.child !== child) return;
        if (!this.closed) {
          if (stderr && !settled) console.error('[CST-MIL startup]', stderr);
          // fail owns the generation guard, so settle promises before clearing it.
          if (this.state.status !== 'error') fail(new Error(`Python inference worker exited (${code ?? 'terminated'}). You can retry the analysis`));
        }
        this.child = null;
      });
    });
  }
  async analyze(session: SessionRecord): Promise<Prediction> {
    await this.start();
    if (this.pending) throw new Error('The inference queue is processing another task');
    const child = this.child;
    if (!child || this.closed) throw new Error('The inference worker is unavailable');
    return new Promise((resolve, reject) => {
      const id = randomUUID();
      const startedAt = performance.now();
      const timer = setTimeout(() => {
        this.pending = null;
        this.state = { ...this.state, status: 'error', error: 'Analysis timed out. Your original input was retained; try again later' };
        if (this.state.firstRequestDurationMs === undefined) { this.state.firstRequestDurationMs = performance.now() - startedAt; this.state.firstRequestStatus = 'failed'; }
        reject(new Error(this.state.error));
        child.kill();
        // A following queued request can start a replacement without waiting for close.
        if (this.child === child) this.child = null;
      }, this.options.requestTimeoutMs ?? 120000);
      this.pending = { id, session, startedAt, resolve, reject, timer };
      child.stdin.write(JSON.stringify({ type: 'analyze', id, session }) + '\n', 'utf8', error => {
        if (this.child !== child) return;
        if (error && this.pending?.id === id) {
          clearTimeout(timer);
          this.pending = null;
          this.state = { status: 'error', error: 'Could not send the inference request' };
          reject(new Error(this.state.error));
          child.kill();
        }
      });
    });
  }
  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    this.failStartup?.(new Error('The inference service is shutting down; the task was interrupted'));
    const child = this.child;
    if (!child) return;
    await new Promise<void>(resolve => {
      const timer = setTimeout(() => { child.kill(); resolve(); }, 1500);
      child.once('close', () => { clearTimeout(timer); resolve(); });
      if (child.stdin.writable) child.stdin.end(JSON.stringify({ type: 'shutdown' }) + '\n');
      else child.kill();
    });
    this.child = null;
  }
}
