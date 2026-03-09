import api from './client'

export interface DatasetListItem {
  id: string
  name: string
  source_type: string
  has_snapshots: boolean
  has_trajectories: boolean
  snapshot_row_count: number | null
  trajectory_row_count: number | null
  registered_at: string
}

export interface DatasetDetail extends DatasetListItem {
  version: number
  content_hash: string
  source_path: string | null
  snapshot_column_count: number | null
  trajectory_column_count: number | null
  snapshot_columns: ColumnInfo[] | null
  trajectory_columns: ColumnInfo[] | null
}

export interface ColumnInfo {
  name: string
  dtype: string
  is_numeric: boolean
  is_categorical: boolean
  null_rate: number
  unique_count: number
  mean?: number | null
  std?: number | null
  min?: number | null
  max?: number | null
}

export interface TargetInfo {
  column: string
  task_type: string
  is_default: boolean
  null_rate?: number
  unique_count?: number
  classes?: string[]
  class_balance?: Record<string, number>
}

export interface AvailableTargets {
  dataset_id: string
  time_series_targets: TargetInfo[]
  entity_targets: TargetInfo[]
}

export interface PreviewData {
  dataset_id: string
  layer: string
  columns: string[]
  rows: Record<string, unknown>[]
  total_rows: number
}

export const fetchDatasets = () =>
  api.get<DatasetListItem[]>('/datasets').then(r => r.data)

export const fetchDataset = (id: string) =>
  api.get<DatasetDetail>(`/datasets/${id}`).then(r => r.data)

export const fetchSchema = (id: string, layer = 'snapshots') =>
  api.get<{ columns: ColumnInfo[] }>(`/datasets/${id}/schema`, { params: { layer } }).then(r => r.data)

export const fetchTargets = (id: string) =>
  api.get<AvailableTargets>(`/datasets/${id}/available-targets`).then(r => r.data)

export const fetchPreview = (id: string, layer = 'snapshots', rows = 10) =>
  api.get<PreviewData>(`/datasets/${id}/preview`, { params: { layer, rows } }).then(r => r.data)

export const uploadDataset = (file: File, name?: string) => {
  const form = new FormData()
  form.append('file', file)
  if (name) form.append('name', name)
  return api.post('/datasets/', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then(r => r.data)
}
