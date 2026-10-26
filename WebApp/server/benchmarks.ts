import type { Benchmarks } from '../shared/types.ts';

export async function getBenchmarks(): Promise<Benchmarks> {
  return {
    dataset: 'Source datasets are obtained separately',
    split: 'Prepare partitions locally using the research code',
    note: 'Raw predictions, evaluation tables, training logs and historical experiment summaries are not included in this public repository.',
    sources: [],
    methods: [],
  };
}
