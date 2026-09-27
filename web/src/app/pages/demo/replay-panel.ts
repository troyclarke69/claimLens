import { ChangeDetectionStrategy, Component, computed, inject, input, output, signal } from '@angular/core';
import { PercentPipe } from '@angular/common';
import { ReplayService } from '../../core/replay.service';
import { FieldName } from '../../core/models';
import { DocumentViewer } from '../../shared/document-viewer';
import { ExtractionResult } from '../../shared/extraction-result';

@Component({
  selector: 'app-replay-panel',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [DocumentViewer, ExtractionResult, PercentPipe],
  templateUrl: './replay-panel.html',
  styleUrl: './replay-panel.css',
})
export class ReplayPanel {
  private readonly replay = inject(ReplayService);

  /** From the URL; fall back to the first document / production model. */
  readonly docId = input<string>();
  readonly modelKey = input<string>();
  readonly selectionChange = output<{ doc: string; model: string }>();

  protected readonly bundle = this.replay.bundle;
  protected readonly showGold = signal(false);
  protected readonly selectedField = signal<FieldName | null>(null);

  protected readonly docs = computed(() => this.bundle.value()?.docs ?? []);
  protected readonly models = computed(() => Object.entries(this.bundle.value()?.models ?? {}).map(([key, label]) => ({ key, label })));

  protected readonly doc = computed(() => {
    const docs = this.docs();
    return docs.find((d) => d.doc_id === this.docId()) ?? docs[0];
  });
  protected readonly model = computed(() => {
    const keys = this.models().map((m) => m.key);
    const k = this.modelKey();
    return k && keys.includes(k) ? k : 'sft_v2';
  });
  protected readonly prediction = computed(() => this.doc()?.predictions[this.model()]);

  protected readonly viewerDoc = computed(() => {
    const d = this.doc();
    return d ? { src: d.image, width: d.width, height: d.height } : null;
  });

  protected pick(doc: string | undefined, model: string | undefined) {
    this.selectedField.set(null);
    this.selectionChange.emit({ doc: doc ?? this.doc()!.doc_id, model: model ?? this.model() });
  }

  protected docTypeLabel(t: string): string {
    return t.startsWith('invoice') ? 'Invoice' : 'Claim form';
  }
}
