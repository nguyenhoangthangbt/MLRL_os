import { useEffect, useState } from 'react'
import { useBuilderStore } from '../../store/builder'
import { fetchPreview, type PreviewData } from '../../api/datasets'

export default function StepExplore() {
  const { config, prevStep, nextStep } = useBuilderStore()
  const [preview, setPreview] = useState<PreviewData | null>(null)

  useEffect(() => {
    if (!config.dataset_id) return
    fetchPreview(config.dataset_id, 'snapshots', 10)
      .then(setPreview)
      .catch(() =>
        fetchPreview(config.dataset_id!, 'trajectories', 10)
          .then(setPreview)
          .catch(() => {})
      )
  }, [config.dataset_id])

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">Step 2: Explore Data</h3>
      <p className="text-sm text-gray-500">Preview your dataset before configuring the experiment.</p>

      {preview ? (
        <div className="overflow-x-auto">
          <p className="text-xs text-gray-400 mb-2">
            {preview.layer} — {preview.total_rows} total rows, showing first {preview.rows.length}
          </p>
          <table className="text-xs w-full border">
            <thead>
              <tr className="bg-gray-50 border-b">
                {preview.columns.map(c => (
                  <th key={c} className="px-2 py-1 text-left font-medium">{c}</th>
                ))}
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
      ) : (
        <p className="text-sm text-gray-400">Loading preview...</p>
      )}

      <div className="flex justify-between pt-4">
        <button onClick={prevStep} className="px-5 py-2 border text-sm rounded-lg hover:bg-gray-50">Back</button>
        <button onClick={nextStep} className="px-5 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700">Next</button>
      </div>
    </div>
  )
}
