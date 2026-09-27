/** Shapes shared by the replay bundle and the live API (POST /v1/extract). */

export const FIELD_NAMES = [
  'claim_number',
  'policy_number',
  'claimant_name',
  'date_of_loss',
  'provider_name',
  'incident_type',
  'total_amount',
] as const;

export type FieldName = (typeof FIELD_NAMES)[number];

export const FIELD_LABELS: Record<FieldName, string> = {
  claim_number: 'Claim number',
  policy_number: 'Policy number',
  claimant_name: 'Claimant name',
  date_of_loss: 'Date of loss',
  provider_name: 'Provider',
  incident_type: 'Incident type',
  total_amount: 'Total amount',
};

/** [x0, y0, x1, y1] in page pixels. */
export type BBox = [number, number, number, number];

export interface ExtractedField {
  value: string;
  evidence_text: string;
  page: number;
  bbox: BBox | null;
}

/** Exactly what the FastAPI service returns. */
export interface ExtractResponse {
  request_id: string;
  fields: Record<FieldName, ExtractedField | null>;
  review: { needs_review: boolean; reasons: string[] };
  model: string;
  latency_ms: number;
  audit: { record_hash: string; prev_hash: string };
}

export interface FieldScore {
  error_type: string;
  correct: boolean;
  citation_ok: boolean | null;
  iou: number | null;
}

export interface ReplayPrediction {
  response: ExtractResponse;
  raw_output: string;
  model_revision: string;
  scores: {
    field_accuracy: number;
    grounded_accuracy: number;
    all_correct: boolean;
    fields: Record<FieldName, FieldScore>;
  };
}

export interface ReplayDoc {
  doc_id: string;
  split: 'test' | 'holdout';
  doc_type: string;
  note: string;
  image: string;
  width: number;
  height: number;
  gold: Record<FieldName, ExtractedField | null>;
  predictions: Record<string, ReplayPrediction>;
}

export interface RunMetrics {
  run_id: string;
  evaluator_version: string;
  field_accuracy: number;
  grounded_accuracy: number;
  hallucination_rate: number;
  miss_rate: number;
  doc_exact_match: number;
  latency_ms_p50: number;
  counterfactual_consistency?: number;
}

export interface ReplayBundle {
  models: Record<string, string>;
  production: string;
  metrics: Record<string, Partial<Record<'test' | 'holdout' | 'fairness', RunMetrics | null>>>;
  docs: ReplayDoc[];
}

/** What the viewer needs to draw a document with its citations. */
export interface ViewerDoc {
  src: string;
  width: number;
  height: number;
}
