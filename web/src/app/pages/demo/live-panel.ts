import { ChangeDetectionStrategy, Component, DestroyRef, computed, inject, signal } from '@angular/core';
import { ExtractApiService, ModelInfo, describeApiError } from '../../core/extract-api.service';
import { ReplayService } from '../../core/replay.service';
import { ExtractResponse, FieldName, ViewerDoc } from '../../core/models';
import { DocumentViewer } from '../../shared/document-viewer';
import { ExtractionResult } from '../../shared/extraction-result';

type Conn = { state: 'idle' | 'checking' } | { state: 'ok'; model: ModelInfo; version: string } | { state: 'error'; message: string };

@Component({
  selector: 'app-live-panel',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [DocumentViewer, ExtractionResult],
  templateUrl: './live-panel.html',
  styleUrl: './live-panel.css',
})
export class LivePanel {
  protected readonly api = inject(ExtractApiService);
  private readonly replay = inject(ReplayService);

  /** This site's origin, which the service must allow (CORS). */
  protected readonly origin = location.origin;

  protected readonly conn = signal<Conn>({ state: 'idle' });
  protected readonly connected = computed(() => {
    const c = this.conn();
    return c.state === 'ok' ? c : null;
  });
  protected readonly connError = computed(() => {
    const c = this.conn();
    return c.state === 'error' ? c.message : null;
  });
  protected readonly simulated = computed(() => this.connected()?.model.name === 'simulated-predictor');

  protected readonly samples = computed(() => this.replay.bundle.value()?.docs ?? []);
  protected readonly source = signal<'sample' | 'upload'>('sample');
  protected readonly sampleId = signal<string>('test-00041');
  protected readonly file = signal<File | null>(null);
  protected readonly docId = signal('');

  protected readonly busy = signal(false);
  protected readonly error = signal<string | null>(null);
  protected readonly result = signal<ExtractResponse | null>(null);
  protected readonly shown = signal<ViewerDoc | null>(null);
  protected readonly selectedField = signal<FieldName | null>(null);

  /** Ground truth is only known for the sample documents. */
  protected readonly gold = computed(() => {
    if (this.source() !== 'sample') return null;
    return this.samples().find((d) => d.doc_id === this.sampleId())?.gold ?? null;
  });

  protected readonly canSend = computed(() => {
    if (this.busy() || this.conn().state !== 'ok') return false;
    if (this.source() === 'upload') return !!this.file() && (!this.simulated() || !!this.docId().trim());
    return true;
  });

  private objectUrl: string | null = null;

  constructor() {
    inject(DestroyRef).onDestroy(() => this.revoke());
  }

  async connect() {
    this.conn.set({ state: 'checking' });
    try {
      const [health, model] = await Promise.all([this.api.health(), this.api.model()]);
      this.conn.set({ state: 'ok', model, version: health.version });
    } catch (e) {
      this.conn.set({ state: 'error', message: describeApiError(e) });
    }
  }

  protected onFile(ev: Event) {
    const f = (ev.target as HTMLInputElement).files?.[0] ?? null;
    this.file.set(f);
    const m = f?.name.match(/^((?:test|holdout|val|train|fairness)-\d{5})/);
    if (m) this.docId.set(m[1]); // dataset images carry their doc_id in the file name
  }

  async send() {
    this.busy.set(true);
    this.error.set(null);
    this.result.set(null);
    this.selectedField.set(null);
    try {
      let blob: Blob;
      let name: string;
      let docId: string | undefined;
      if (this.source() === 'sample') {
        const s = this.samples().find((d) => d.doc_id === this.sampleId())!;
        blob = await (await fetch(s.image)).blob();
        name = `${s.doc_id}.jpg`;
        docId = s.doc_id;
      } else {
        blob = this.file()!;
        name = this.file()!.name;
        docId = this.docId().trim() || undefined;
      }
      const shown = await this.preview(blob);
      const res = await this.api.extract(blob, name, docId);
      this.shown.set(shown);
      this.result.set(res);
    } catch (e) {
      this.error.set(describeApiError(e));
    } finally {
      this.busy.set(false);
    }
  }

  /** Object URL + natural size, so boxes are drawn in the image's own pixels. */
  private async preview(blob: Blob): Promise<ViewerDoc> {
    this.revoke();
    this.objectUrl = URL.createObjectURL(blob);
    const img = new Image();
    img.src = this.objectUrl;
    await img.decode();
    return { src: this.objectUrl, width: img.naturalWidth, height: img.naturalHeight };
  }

  private revoke() {
    if (this.objectUrl) URL.revokeObjectURL(this.objectUrl);
    this.objectUrl = null;
  }
}
