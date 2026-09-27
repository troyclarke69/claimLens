import { Injectable } from '@angular/core';
import { httpResource } from '@angular/common/http';
import { ReplayBundle } from './models';

/**
 * Real outputs from the Kaggle runs (see web/scripts/build_replay.py).
 * Loaded once and shared; `httpResource` exposes value / isLoading / error as signals.
 */
@Injectable({ providedIn: 'root' })
export class ReplayService {
  readonly bundle = httpResource<ReplayBundle>(() => 'replay/replay.json');
}
