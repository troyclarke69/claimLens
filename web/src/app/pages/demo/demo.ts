import { ChangeDetectionStrategy, Component, inject, input } from '@angular/core';
import { Router } from '@angular/router';
import { ReplayPanel } from './replay-panel';
import { LivePanel } from './live-panel';

/**
 * /demo?mode=replay|live&doc=…&model=…
 * Query params are bound straight to inputs (withComponentInputBinding), so any
 * view is a shareable link.
 */
@Component({
  selector: 'app-demo',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [ReplayPanel, LivePanel],
  template: `
    <header class="head">
      <div>
        <h1>Demo</h1>
        <p class="muted">
          @if (mode() === 'live') {
            Send a document to a running ClaimLens service (<code>POST /v1/extract</code>) and see what comes back.
          } @else {
            Real outputs from the Kaggle runs, replayed exactly as logged, scored against the ground truth.
          }
        </p>
      </div>
      <div class="seg" role="tablist" aria-label="Demo mode">
        <button role="tab" [attr.aria-selected]="mode() !== 'live'" (click)="setMode('replay')">Replay real runs</button>
        <button role="tab" [attr.aria-selected]="mode() === 'live'" (click)="setMode('live')">Live API</button>
      </div>
    </header>

    @if (mode() === 'live') {
      <app-live-panel />
    } @else {
      <app-replay-panel [docId]="doc()" [modelKey]="model()" (selectionChange)="select($event)" />
    }
  `,
  styles: `
    :host { display: grid; gap: 24px; }
    .head { display: flex; justify-content: space-between; align-items: end; gap: 16px; flex-wrap: wrap; }
    h1 { margin: 0 0 4px; font-size: 1.8rem; letter-spacing: -.01em; }
    .muted { color: var(--ink-2); margin: 0; max-width: 640px; }
    .seg { display: inline-flex; padding: 3px; border-radius: 10px; background: var(--surface-2); border: 1px solid var(--line); }
    .seg button { border: 0; background: transparent; color: var(--ink-2); padding: 8px 14px; border-radius: 8px; font: inherit; font-weight: 600; cursor: pointer; }
    .seg button[aria-selected='true'] { background: var(--surface); color: var(--ink); box-shadow: 0 1px 2px rgb(0 0 0 / .08); }
  `,
})
export class Demo {
  private readonly router = inject(Router);

  readonly mode = input<string>();
  readonly doc = input<string>();
  readonly model = input<string>();

  setMode(mode: 'replay' | 'live') {
    this.router.navigate([], { queryParams: { mode: mode === 'live' ? 'live' : null }, queryParamsHandling: 'merge' });
  }

  select(sel: { doc: string; model: string }) {
    this.router.navigate([], { queryParams: sel, queryParamsHandling: 'merge', replaceUrl: true });
  }
}
