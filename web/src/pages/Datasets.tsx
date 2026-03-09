import { useEffect, useState, useRef } from 'react'
import { Link } from 'react-router-dom'
import Card from '../components/shared/Card'
import { fetchDatasets, uploadDataset, type DatasetListItem } from '../api/datasets'

export default function Datasets() {
  const [datasets, setDatasets] = useState<DatasetListItem[]>([])
  const [uploading, setUploading] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  const load = () => fetchDatasets().then(setDatasets).catch(() => {})

  useEffect(() => { load() }, [])

  const handleUpload = async () => {
    const file = fileRef.current?.files?.[0]
    if (!file) return
    setUploading(true)
    try {
      await uploadDataset(file)
      load()
    } finally {
      setUploading(false)
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-800">Datasets</h2>
        <div className="flex gap-2 items-center">
          <input ref={fileRef} type="file" accept=".csv,.parquet,.json" className="text-sm" />
          <button
            onClick={handleUpload}
            disabled={uploading}
            className="px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            {uploading ? 'Uploading...' : 'Upload'}
          </button>
        </div>
      </div>

      {datasets.length === 0 ? (
        <Card>
          <p className="text-sm text-gray-400">No datasets registered. Upload one above.</p>
        </Card>
      ) : (
        <div className="bg-white rounded-lg shadow-sm border">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-gray-500 border-b bg-gray-50">
                <th className="px-4 py-3">Name</th>
                <th className="px-4 py-3">Source</th>
                <th className="px-4 py-3">Snapshots</th>
                <th className="px-4 py-3">Trajectories</th>
                <th className="px-4 py-3">Registered</th>
              </tr>
            </thead>
            <tbody>
              {datasets.map(d => (
                <tr key={d.id} className="border-b border-gray-50 hover:bg-gray-50">
                  <td className="px-4 py-3">
                    <Link to={`/datasets/${d.id}`} className="text-blue-600 hover:underline">
                      {d.name}
                    </Link>
                  </td>
                  <td className="px-4 py-3 text-gray-500">{d.source_type}</td>
                  <td className="px-4 py-3">{d.snapshot_row_count ?? '-'}</td>
                  <td className="px-4 py-3">{d.trajectory_row_count ?? '-'}</td>
                  <td className="px-4 py-3 text-gray-400">{d.registered_at.slice(0, 19)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
