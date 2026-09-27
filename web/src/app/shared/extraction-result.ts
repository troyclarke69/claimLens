import { ChangeDetectionStrategy, Component, computed, input, model } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { ExtractResponse, ExtractedField, FIELD_LABELS, FIELD_NAMES, FieldName, FieldScore } from '../core/models';
import { verdict } from '../core/verdict';

/** Fields + evidence, the review decision and the audit receipt for one extraction. */
@Component({
  selector: 'app-extraction-result',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [DecimalPipe],
  template: `
    @let r = response();

    <div class="review" [class.flagged]="r.review.needs_review" role="status">
      @if (r.review.needs_review) {
        <strong>Sent to human review</strong>
        <ul>
          @for (reason of r.review.reasons; track reason) {
            <li>{{ reason }}</li>
          }
        </ul>
      } @else {
        <strong>No review flags</strong>
        <span class="sub">The service would pass this straight through{{ scores() ? ', so any error below reaches the claim unchecked' : '' }}.</span>
      }
    </div>

    <table class="fields">
      <thead>
        <tr>
          <th scope="col">Field</th>
          <th scope="col">Value <span class="sub">&amp; evidence</span></th>
          @if (scores()) {
            <th scope="col">Verdict</th>
          }
          @if (gold()) {
            <th scope="col" class="hide-sm">Truth</th>
          }
        </tr>
      </thead>
      <tbody>
        @for (row of rows(); track row.name) {
          <tr
            [class.active]="selected() === row.name"
            (mouseenter)="selected.set(row.name)"
            (mouseleave)="selected.set(null)"
            (focusin)="selected.set(row.name)"
            tabindex="0"
          >
            <th scope="row">{{ row.label }}</th>
            <td>
              @if (row.pred) {
                <span class="value">{{ row.pred.value }}</span>
                @if (row.pred.evidence_text && row.pred.evidence_text !== row.pred.value) {
                  <span class="evidence">“{{ row.pred.evidence_text }}”</span>
                }
                @if (!row.pred.bbox) {
                  <span class="chip tone-warn">no citation</span>
                }
              } @else {
                <span class="null">null</span>
              }
            </td>
            @if (row.verdict; as v) {
              <td>
                <span class="chip" [class]="'chip tone-' + v.tone" [title]="v.help">{{ v.label }}</span>
                @if (row.pred && row.score?.correct && row.score?.citation_ok === false) {
                  <span class="chip tone-warn" title="The value is right but the cited box is not on it (IoU < 0.5).">box off</span>
                }
              </td>
            }
            @if (gold()) {
              <td class="gold hide-sm">{{ row.gold?.value ?? '—' }}</td>
            }
          </tr>
        }
      </tbody>
    </table>

    <dl class="receipt">
      <div>
        <dt>Model</dt>
        <dd>{{ r.model }}</dd>
      </div>
      <div>
        <dt>Latency</dt>
        <dd>{{ r.latency_ms / 1000 | number: '1.1-1' }} s</dd>
      </div>
      <div class="wide">
        <dt>Audit record hash</dt>
        <dd><code>{{ r.audit.record_hash }}</code></dd>
      </div>
      <div class="wide">
        <dt>Previous record hash</dt>
        <dd><code>{{ r.audit.prev_hash }}</code></dd>
      </div>
    </dl>

    @if (rawOutput()) {
      <details class="raw">
        <summary>Raw model output (from the audit log)</summary>
        <pre>{{ rawOutput() }}</pre>
      </details>
    }
  `,
  styles: `
    :host { display: grid; gap: 16px; }
    .review { border-radius: 10px; padding: 12px 14px; background: color-mix(in srgb, var(--good) 10%, var(--surface)); border: 1px solid color-mix(in srgb, var(--good) 35%, transparent); }
    .review.flagged { background: color-mix(in srgb, var(--bad) 9%, var(--surface)); border-color: color-mix(in srgb, var(--bad) 40%, transparent); }
    .review strong { display: block; margin-bottom: 2px; }
    .review ul { margin: 4px 0 0; padding-left: 18px; }
    .sub { color: var(--ink-2); font-weight: 400; font-size: .9em; }
    .fields { width: 100%; border-collapse: collapse; font-size: .92rem; }
    .fields th, .fields td { text-align: left; padding: 8px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
    .fields thead th { font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; color: var(--ink-2); font-weight: 600; }
    .fields tbody th { font-weight: 500; white-space: nowrap; }
    .fields tbody tr { transition: background .12s; outline: none; }
    .fields tbody tr.active, .fields tbody tr:focus-visible { background: var(--hover); }
    .value { font-family: var(--mono); overflow-wrap: break-word; }
    .evidence { display: block; color: var(--ink-2); font-size: .85em; }
    .null { color: var(--ink-3); font-style: italic; }
    .gold { font-family: var(--mono); color: var(--ink-2); overflow-wrap: break-word; }
    .chip { display: inline-block; margin: 0 4px 2px 0; padding: 1px 8px; border-radius: 999px; font-size: .78rem; font-weight: 600; white-space: nowrap; }
    .tone-good { background: color-mix(in srgb, var(--good) 16%, transparent); color: var(--good-ink); }
    .tone-warn { background: color-mix(in srgb, var(--warn) 20%, transparent); color: var(--warn-ink); }
    .tone-bad { background: color-mix(in srgb, var(--bad) 16%, transparent); color: var(--bad-ink); }
    .tone-muted { background: var(--hover); color: var(--ink-2); }
    .receipt { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 16px; margin: 0; padding: 12px 14px; border-radius: 10px; background: var(--surface-2); font-size: .85rem; }
    .receipt .wide { grid-column: 1 / -1; }
    .receipt dt { color: var(--ink-2); font-size: .75rem; text-transform: uppercase; letter-spacing: .04em; }
    .receipt dd { margin: 2px 0 0; }
    .receipt code { font-family: var(--mono); font-size: .8rem; word-break: break-all; }
    .raw summary { cursor: pointer; color: var(--ink-2); font-size: .9rem; }
    .raw pre { max-height: 260px; overflow: auto; padding: 12px; border-radius: 8px; background: var(--surface-2); font-size: .78rem; white-space: pre-wrap; word-break: break-all; }
    @media (max-width: 640px) { .hide-sm { display: none; } }
  `,
})
export class ExtractionResult {
  readonly response = input.required<ExtractResponse>();
  readonly scores = input<Record<FieldName, FieldScore> | null>(null);
  readonly gold = input<Record<FieldName, ExtractedField | null> | null>(null);
  readonly rawOutput = input<string | null>(null);
  readonly selected = model<FieldName | null>(null);

  protected readonly rows = computed(() => {
    const r = this.response();
    const scores = this.scores();
    const gold = this.gold();
    return FIELD_NAMES.map((name) => ({
      name,
      label: FIELD_LABELS[name],
      pred: r.fields[name] ?? null,
      gold: gold?.[name] ?? null,
      score: scores?.[name] ?? null,
      verdict: scores ? verdict(scores[name]?.error_type) : null,
    }));
  });
}
