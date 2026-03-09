import api from './client'

export interface ModelListItem {
  id: string
  experiment_id: string
  algorithm_name: string
  task: string
  metrics: Record<string, number>
  created_at: string
}

export interface ModelDetail extends ModelListItem {
  feature_names: string[]
  file_path: string
}

export interface PredictResponse {
  model_id: string
  predictions: (number | string)[]
  task_type: string
  probabilities: Record<string, number>[] | null
}

export const fetchModels = () =>
  api.get<ModelListItem[]>('/models').then(r => r.data)

export const fetchModel = (id: string) =>
  api.get<ModelDetail>(`/models/${id}`).then(r => r.data)

export const predict = (id: string, data: Record<string, unknown>[]) =>
  api.post<PredictResponse>(`/models/${id}/predict`, { data }).then(r => r.data)

export const fetchFeatureImportance = (id: string) =>
  api.get(`/models/${id}/feature-importance`).then(r => r.data)
