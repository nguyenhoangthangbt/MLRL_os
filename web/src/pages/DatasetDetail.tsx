import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import Card from '../components/shared/Card'
import { fetchDataset, fetchPreview, type DatasetDetail as DDetail, type PreviewData } from '../api/datasets'

export default function DatasetDetail() {
  const { id } = useParams<{ id: string }>()
  const [dataset, setDataset] = useState<DDetail | null>(null)
  const [preview, setPreview] = useState<PreviewData | null>(null)

  useEffect(() => {
    if (!id) return
    fetchDataset(id).then(setDataset).catch(() => {})
    fetchPreview(id, 'snapshots', 10)
      .then(setPreview)
      .catch(() => fetchPreview(id, 'trajectories', 10).then(setPreview).catch(() => {}))
  }, [id])

  if (!dataset) return <p className="text-gray-400">Loading...</p>

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-800">{dataset.name}</h2>

      <div className="grid grid-cols-2 gap-4">
        <Card title="Info">
          <dl className="space-y-2 text-sm">
            <div className="flex justify-between"><dt className="text-gray-500">ID</dt><dd className="font-mono">{dataset.id}</dd></div>
            <div className="flex justify-between"><dt className="text-gray-500">Source</dt><dd>{dataset.source_type}</dd></div>
            <div className="flex justify-between"><dt className="text-gray-500">Snapshots</dt><dd>{dataset.snapshot_row_count ?? '-'} rows</dd></div>
            <div className="flex justify-between"><dt className="text-gray-500">Trajectories</dt><dd>{dataset.trajectory_row_count ?? '-'} rows</dd></div>
          </dl>
        </Card>
        <Card title="Columns">
          {dataset.snapshot_columns ? (
            <ul className="text-sm space-y-1 max-h-48 overflow-y-auto">
              {dataset.snapshot_columns.map(c => (
                <li key={c.name} className="flex justify-between">
                  <span className="font-mono">{c.name}</span>
                  <span className="text-gray-400">{c.dtype}</span>
                </li>
              ))}
            </ul>
          ) : dataset.trajectory_columns ? (
            <ul className="text-sm space-y-1 max-h-48 overflow-y-auto">
              {dataset.trajectory_columns.map(c => (
                <li key={c.name} className="flex justify-between">
                  <span className="font-mono">{c.name}</span>
                  <span className="text-gray-400">{c.dtype}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-gray-400 text-sm">No column info</p>
          )}
        </Card>
      </div>

      {preview && (
        <Card title={`Preview (${preview.layer}) — ${preview.total_rows} total rows`}>
          <div className="overflow-x-auto">
            <table className="text-xs w-full">
              <thead>
                <tr className="border-b text-gray-500">
                  {preview.columns.map(c => <th key={c} className="px-2 py-1 text-left">{c}</th>)}
                </tr>
              </thead>
              <tbody>
                {preview.rows.map((row, i) => (
                  <tr key={i} className="border-b border-gray-50">
                    {preview.columns.map(c => (
                      <td key={c} className="px-2 py-1 font-mono">{String(row[c] ?? '')}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  )
}
