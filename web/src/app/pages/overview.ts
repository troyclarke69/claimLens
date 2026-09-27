import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { PercentPipe } from '@angular/common';
import { RouterLink } from '@angular/router';
import { ReplayService } from '../core/replay.service';
import { RunMetrics } from '../core/models';

type Col = { key: keyof RunMetrics; split: 'test' | 'holdout'; label: string; better: 'high' | 'low' };

const COLUMNS: Col[] = [
  { split: 'test', key: 'grounded_accuracy', label: 'Grounded', better: 'high' },
  { split: 'test', key: 'hallucination_rate', label: 'Hallucination', better: 'low' },
  { split: 'holdout', key: 'field_accuracy', label: 'Field accuracy', better: 'high' },
  { split: 'holdout', key: 'grounded_accuracy', label: 'Grounded', better: 'high' },
  { split: 'holdout', key: 'doc_exact_match', label: 'Exact match', better: 'high' },
  { split: 'holdout', key: 'hallucination_rate', label: 'Hallucination', better: 'low' },
  { split: 'holdout', key: 'miss_rate', label: 'Miss', better: 'low' },
];

const ROWS = [
  { key: 'baseline', label: 'Baseline', detail: 'prompt only' },
  { key: 'sft_v1', label: 'SFT v1', detail: '300 docs, 2 layouts' },
  { key: 'sft_v2ctrl', label: 'SFT v2', detail: '+80 randomised-layout docs' },
  { key: 'grpo_v1', label: 'GRPO', detail: 'same 80 docs, cost-weighted reward' },
];

@Component({
  selector: 'app-overview',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [RouterLink, PercentPipe],
  templateUrl: './overview.html',
  styleUrl: './overview.css',
})
export class Overview {
  private readonly replay = inject(ReplayService);
  protected readonly columns = COLUMNS;
  protected readonly loading = this.replay.bundle.isLoading;

  /** Results table, straight from models/registry.json (evaluator 1.1). */
  protected readonly table = computed(() => {
    const metrics = this.replay.bundle.value()?.metrics;
    if (!metrics) return null;
    const cell = (row: string, c: Col) => (metrics[row]?.[c.split]?.[c.key] as number | undefined) ?? null;
    const best = COLUMNS.map((c) => {
      const vals = ROWS.map((r) => cell(r.key, c)).filter((v): v is number => v !== null);
      return c.better === 'high' ? Math.max(...vals) : Math.min(...vals);
    });
    return ROWS.map((r) => ({
      ...r,
      production: r.key === this.replay.bundle.value()?.production,
      cells: COLUMNS.map((c, i) => ({ value: cell(r.key, c), best: cell(r.key, c) === best[i] })),
    }));
  });
}
