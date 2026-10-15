import type { InferenceEngine } from './worker.ts';
import { Store } from './store.ts';
import { workerIdentity } from './prediction.ts';

export class AnalysisQueue {
  private waiting: string[] = [];
  private processing: Promise<void> | null = null;
  private closed = false;
  constructor(private store: Store, private engine: InferenceEngine) {}
  enqueue(ids: string[]) {
    if (this.closed) throw new Error('服务正在关闭');
    this.waiting.push(...ids);
    if (!this.processing) this.processing = this.drain().finally(() => { this.processing = null; });
  }
  private async drain() {
    while (this.waiting.length && !this.closed) {
      const id = this.waiting.shift()!;
      const record = this.store.getAnalysis(id);
      this.store.markRunning(id);
      const started = performance.now();
      try {
        const session = this.store.getSession(record.sessionId);
        const result = await this.engine.analyze(session);
        if (this.closed) this.store.fail(id, '服务关闭导致任务中断，请重新分析');
        else this.store.succeed(id, result, workerIdentity(this.engine.health()), performance.now() - started);
      } catch (error) {
        this.store.fail(id, error instanceof Error ? error.message : '分析失败，请重试');
      }
    }
  }
  async close() {
    this.closed = true;
    for (const id of this.waiting.splice(0)) this.store.fail(id, '服务关闭导致等待任务中断，请重新分析');
    await this.engine.close();
    await this.processing;
  }
}
