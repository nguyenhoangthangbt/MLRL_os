import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { fetchModels, type ModelListItem } from '../api/models'

export default function Models() {
  const [models, setModels] = useState<ModelListItem[]>([])

  useEffect(() => { fetchModels().then(setModels).catch(() => {}) }, [])

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-800">Models</h2>

      {models.length === 0 ? (
        <p className="text-sm text-gray-400">No models registered yet. Run an experiment to train models.</p>
      ) : (
        <div className="bg-white rounded-lg shadow-sm border">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-gray-500 border-b bg-gray-50">
                <th className="px-4 py-3">ID</th>
                <th className="px-4 py-3">Algorithm</th>
                <th className="px-4 py-3">Task</th>
                <th className="px-4 py-3">Metrics</th>
                <th className="px-4 py-3">Experiment</th>
                <th className="px-4 py-3">Created</th>
              </tr>
            </thead>
            <tbody>
              {models.map(m => (
                <tr key={m.id} className="border-b border-gray-50 hover:bg-gray-50">
                  <td className="px-4 py-3 font-mono text-xs">{m.id}</td>
                  <td className="px-4 py-3">{m.algorithm_name}</td>
                  <td className="px-4 py-3 text-gray-500">{m.task}</td>
                  <td className="px-4 py-3 text-xs font-mono">
                    {Object.entries(m.metrics).slice(0, 2).map(([k, v]) => `${k}=${v.toFixed(4)}`).join(', ')}
                  </td>
                  <td className="px-4 py-3">
                    <Link to={`/experiments/${m.experiment_id}`} className="text-blue-600 hover:underline text-xs font-mono">
                      {m.experiment_id}
                    </Link>
                  </td>
                  <td className="px-4 py-3 text-gray-400">{m.created_at.slice(0, 19)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
