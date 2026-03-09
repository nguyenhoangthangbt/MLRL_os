import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import StatusBadge from '../components/shared/StatusBadge'
import { fetchExperiments, type ExperimentListItem } from '../api/experiments'

export default function Experiments() {
  const [experiments, setExperiments] = useState<ExperimentListItem[]>([])

  useEffect(() => { fetchExperiments().then(setExperiments).catch(() => {}) }, [])

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-800">Experiments</h2>
        <Link to="/builder" className="px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700">
          New Experiment
        </Link>
      </div>

      {experiments.length === 0 ? (
        <p className="text-sm text-gray-400">No experiments yet.</p>
      ) : (
        <div className="bg-white rounded-lg shadow-sm border">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-gray-500 border-b bg-gray-50">
                <th className="px-4 py-3">Name</th>
                <th className="px-4 py-3">Type</th>
                <th className="px-4 py-3">Status</th>
                <th className="px-4 py-3">Best Algorithm</th>
                <th className="px-4 py-3">Duration</th>
                <th className="px-4 py-3">Created</th>
              </tr>
            </thead>
            <tbody>
              {experiments.map(e => (
                <tr key={e.experiment_id} className="border-b border-gray-50 hover:bg-gray-50">
                  <td className="px-4 py-3">
                    <Link to={`/experiments/${e.experiment_id}`} className="text-blue-600 hover:underline">
                      {e.name}
                    </Link>
                  </td>
                  <td className="px-4 py-3 text-gray-500">{e.experiment_type}</td>
                  <td className="px-4 py-3"><StatusBadge status={e.status} /></td>
                  <td className="px-4 py-3">{e.best_algorithm ?? '-'}</td>
                  <td className="px-4 py-3 text-gray-500">{e.duration_seconds ? `${e.duration_seconds.toFixed(1)}s` : '-'}</td>
                  <td className="px-4 py-3 text-gray-400">{e.created_at.slice(0, 19)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
