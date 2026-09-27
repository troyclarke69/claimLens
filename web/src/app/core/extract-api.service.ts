import { Injectable, effect, inject, signal } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { firstValueFrom, timeout } from 'rxjs';
import { ExtractResponse } from './models';

export interface ModelInfo {
  name: string;
  note?: string;
  base_model?: string;
  model_revision?: string | null;
  prompt_version: string;
  evaluator_version: string;
  code_version: string;
}

const STORAGE_KEY = 'claimlens.apiBase';

function readStored(): string | null {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null; // private mode, blocked storage, ...
  }
}

/** Thin client for the FastAPI service in claimlens/service.py. */
@Injectable({ providedIn: 'root' })
export class ExtractApiService {
  private readonly http = inject(HttpClient);

  /** Where the service runs. Remembered per browser. */
  readonly baseUrl = signal(readStored() ?? 'http://localhost:8000');

  constructor() {
    effect(() => {
      const url = this.baseUrl();
      try {
        localStorage.setItem(STORAGE_KEY, url);
      } catch {
        /* not critical */
      }
    });
  }

  private url(path: string): string {
    return this.baseUrl().replace(/\/+$/, '') + path;
  }

  health(): Promise<{ status: string; version: string }> {
    return firstValueFrom(this.http.get<{ status: string; version: string }>(this.url('/v1/health')).pipe(timeout(5000)));
  }

  model(): Promise<ModelInfo> {
    return firstValueFrom(this.http.get<ModelInfo>(this.url('/v1/model')).pipe(timeout(5000)));
  }

  verifyAudit(): Promise<{ ok: boolean; message: string }> {
    return firstValueFrom(this.http.get<{ ok: boolean; message: string }>(this.url('/v1/audit/verify')));
  }

  /** Multipart upload, same as the Swagger page at /docs. */
  extract(file: Blob, filename: string, docId?: string): Promise<ExtractResponse> {
    const form = new FormData();
    form.append('file', file, filename);
    if (docId) form.append('doc_id', docId);
    return firstValueFrom(this.http.post<ExtractResponse>(this.url('/v1/extract'), form).pipe(timeout(180_000)));
  }
}

/** Turn an HTTP failure into a sentence a visitor can act on. */
export function describeApiError(err: unknown): string {
  if (err instanceof HttpErrorResponse) {
    if (err.status === 0) {
      return `Could not reach the service. Is it running, and does CLAIMLENS_CORS_ORIGINS include ${location.origin}?`;
    }
    const detail = typeof err.error === 'object' && err.error?.detail ? err.error.detail : err.message;
    return `${err.status}: ${typeof detail === 'string' ? detail : JSON.stringify(detail)}`;
  }
  if (err instanceof Error && err.name === 'TimeoutError') return 'The service did not answer in time.';
  return String(err);
}
