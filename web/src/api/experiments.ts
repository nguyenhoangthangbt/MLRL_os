import api from './client'

export interface ExperimentRequest {
  dataset_id: string
  name?: string
  seed?: number
  dataset_layer?: string
  experiment_type?: string
  target?: string
  lookback?: string
  horizon?: string
  lag_intervals?: string[]
  rolling_windows?: string[]
  observation_point?: string
  feature_columns?: string[]
  exclude_columns?: string[]
  algorithms?: string[]
  cv_folds?: number
  cv_strategy?: string
  metrics?: string[]
  handle_imbalance?: boolean
  hyperparameter_tuning?: boolean
  generate_report?: boolean
}

export interface ExperimentListItem {
  experiment_id: string
  name: string
  status: string
  experiment_type: string
  created_at: string
  completed_at: string | null
  best_algorithm: string | null
  duration_seconds: number | null
}

export interface ExperimentDetail extends ExperimentListItem {
  metrics: Record<string, number> | null
  all_algorithm_scores: AlgorithmScore[] | null
  feature_importance: FeatureImportanceEntry[] | null
  model_id: string | null
  sample_count: number | null
  feature_count: number | null
  resolved_config: Record<string, unknown> | null
  error_message: string | null
}

export interface AlgorithmScore {
  algorithm: string
  metrics: Record<string, number>
  metrics_std: Record<string, number>
  rank: number
}

export interface FeatureImportanceEntry {
  feature: string
  importance: number
  rank: number
}

export interface ValidationResponse {
  valid: boolean
  resolved_config: Record<string, unknown> | null
  errors: { code: string; field: string; message: string; suggestion?: string }[]
  warnings: string[]
}

export interface DefaultsResponse {
  problem_type: string
  defaults: Record<string, unknown>
}

export const fetchExperiments = () =>
  api.get<ExperimentListItem[]>('/experiments').then(r => r.data)

export const fetchExperiment = (id: string) =>
  api.get<ExperimentDetail>(`/experiments/${id}`).then(r => r.data)

export const fetchReport = (id: string) =>
  api.get(`/experiments/${id}/report`).then(r => r.data)

export const submitExperiment = (req: ExperimentRequest) =>
  api.post('/experiments/', req).then(r => r.data)

export const validateExperiment = (req: ExperimentRequest) =>
  api.post<ValidationResponse>('/experiments/validate', req).then(r => r.data)

export const fetchDefaults = (problemType: string) =>
  api.get<DefaultsResponse>(`/experiments/defaults/${problemType}`).then(r => r.data)
