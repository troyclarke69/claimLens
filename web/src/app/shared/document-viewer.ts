import { ChangeDetectionStrategy, Component, computed, input, model } from '@angular/core';
import { BBox, ExtractedField, FIELD_LABELS, FIELD_NAMES, FieldName, FieldScore, ViewerDoc } from '../core/models';
import { Tone, verdict } from '../core/verdict';

interface Box {
  field: FieldName;
  bbox: BBox;
  kind: 'pred' | 'gold';
  tone: Tone | 'accent';
  label: string;
}

/**
 * The page image with the model's citation boxes drawn on top.
 * The SVG uses the page's own pixel coordinates as its viewBox, so boxes line
 * up at any display size without any maths in the template.
 */
@Component({
  selector: 'app-document-viewer',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <figure class="viewer" [style.aspect-ratio]="doc().width + ' / ' + doc().height">
      <img [src]="doc().src" [alt]="'Scanned document ' + (alt() ?? '')" />
      <svg [attr.viewBox]="'0 0 ' + doc().width + ' ' + doc().height" preserveAspectRatio="none" aria-hidden="true">
        @for (b of boxes(); track b.kind + b.field) {
          <rect
            [attr.x]="b.bbox[0] - 3"
            [attr.y]="b.bbox[1] - 3"
            [attr.width]="b.bbox[2] - b.bbox[0] + 6"
            [attr.height]="b.bbox[3] - b.bbox[1] + 6"
            [class]="'box ' + b.kind + ' tone-' + b.tone"
            [class.active]="selected() === b.field"
            [class.dim]="selected() !== null && selected() !== b.field"
            (click)="selected.set(selected() === b.field ? null : b.field)"
          >
            <title>{{ b.label }}</title>
          </rect>
        }
      </svg>
    </figure>
  `,
  styles: `
    .viewer {
      position: relative;
      margin: 0;
      width: 100%;
      border-radius: 10px;
      overflow: hidden;
      border: 1px solid var(--line);
      background: #fff;
    }
    img, svg { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
    .box { fill: transparent; stroke-width: 3; cursor: pointer; transition: opacity .15s, stroke-width .15s; vector-effect: non-scaling-stroke; }
    .box.pred { stroke-width: 2.5; }
    .box.gold { stroke: var(--ink-2); stroke-dasharray: 5 4; stroke-width: 1.5; pointer-events: none; }
    .box.pred.tone-good { stroke: var(--good); fill: color-mix(in srgb, var(--good) 12%, transparent); }
    .box.pred.tone-warn { stroke: var(--warn); fill: color-mix(in srgb, var(--warn) 14%, transparent); }
    .box.pred.tone-bad { stroke: var(--bad); fill: color-mix(in srgb, var(--bad) 14%, transparent); }
    .box.pred.tone-accent, .box.pred.tone-muted { stroke: var(--accent); fill: color-mix(in srgb, var(--accent) 12%, transparent); }
    .box.active { stroke-width: 4; }
    .box.dim { opacity: .25; }
  `,
})
export class DocumentViewer {
  readonly doc = input.required<ViewerDoc>();
  readonly fields = input<Record<FieldName, ExtractedField | null> | null>(null);
  /** When given, boxes are coloured by the evaluator's verdict. */
  readonly scores = input<Record<FieldName, FieldScore> | null>(null);
  /** Ground-truth boxes, drawn dashed. */
  readonly gold = input<Record<FieldName, ExtractedField | null> | null>(null);
  readonly alt = input<string>();
  /** Two-way bound with the fields table: hover/click either side to highlight. */
  readonly selected = model<FieldName | null>(null);

  protected readonly boxes = computed<Box[]>(() => {
    const out: Box[] = [];
    const gold = this.gold();
    const fields = this.fields();
    const scores = this.scores();
    for (const f of FIELD_NAMES) {
      const g = gold?.[f];
      if (g?.bbox) out.push({ field: f, bbox: g.bbox, kind: 'gold', tone: 'muted', label: `${FIELD_LABELS[f]} (ground truth)` });
    }
    for (const f of FIELD_NAMES) {
      const p = fields?.[f];
      if (!p?.bbox) continue;
      const v = scores ? verdict(scores[f]?.error_type) : null;
      const tone: Box['tone'] = v ? (v.tone === 'good' && scores?.[f]?.citation_ok === false ? 'warn' : v.tone) : 'accent';
      out.push({ field: f, bbox: p.bbox, kind: 'pred', tone, label: `${FIELD_LABELS[f]}: ${p.value}` });
    }
    return out;
  });
}
