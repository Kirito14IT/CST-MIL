import { readFile } from 'node:fs/promises';
import path from 'node:path';
import type { BenchmarkRow, Benchmarks } from '../shared/types.ts';
import { FOLLOWUP_ROOT } from './paths.ts';
import { HttpError } from './validation.ts';

export async function getBenchmarks(): Promise<Benchmarks> {
  const names = ['performance.json', 'evidence.json', 'costs.json'];
  type Metrics = { method: string } & Record<string, string | number>;
  let values: Metrics[][];
  try { values = await Promise.all(names.map(async name => JSON.parse(await readFile(path.join(FOLLOWUP_ROOT, name), 'utf8')) as Metrics[])); }
  catch { throw new HttpError(503, '历史基准文件不可用，请保留 cst_mil/benchmark/research_followup/20260922 目录'); }
  const labels: Record<string, string> = { lda: 'LDA', btm: 'BTM', gsdmm: 'GSDMM', nmf: 'NMF', bertopic: 'BERTopic', cst_mil_r0: 'CST-MIL' };
  const fields = ['macro_f1', 'recall_cb', 'precision_cb', 'ap', 'roc_auc', 'hit_rate_at_3', 'mean_recall_at_3', 'comment_ap', 'train_seconds', 'inference_test_seconds', 'latency_mean_seconds', 'peak_rss_mib', 'model_mib'] as const;
  const methods = Object.entries(labels).map(([method, label]) => {
    const source = Object.assign({}, ...values.map(rows => rows.find(row => row.method === method)));
    const metrics: Record<string, number> = {};
    for (const field of fields) {
      if (typeof source[field] !== 'number' || !Number.isFinite(source[field])) throw new HttpError(503, `历史基准指标缺失：${label} / ${field}`);
      metrics[field] = source[field];
    }
    return { method, label, ...metrics } as BenchmarkRow;
  });
  return { dataset: 'SCCD：677 个微博完整会话', split: '训练 406 / 验证 135 / 测试 136；证据命中分母为 101 个含标注危险评论的测试会话', note: '本页为 2026-09-22 对已有冻结实验的复核，属于微博数据历史结果，不代表校园场景效果。Top-3 是人工核查候选；CST-MIL 对 NMF 的 AP/AUC 差值 95% 区间跨零，不能据此声称稳定优于全部基线。时间为历史测试流程记录，不是当前电脑实时测试；模型大小不含共享 Python 环境，BERTopic 编码器及 GSDMM Java 运行时另计。', sources: names.map(name => `cst_mil/benchmark/research_followup/20260922/${name}`), methods };
}
