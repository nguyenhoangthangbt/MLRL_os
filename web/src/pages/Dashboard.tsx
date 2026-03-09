import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import Card from '../components/shared/Card'
import StatusBadge from '../components/shared/StatusBadge'
import { fetchDatasets, type DatasetListItem } from '../api/datasets'
import { fetchExperiments, type ExperimentListItem } from '../api/experiments'

export default function Dashboard() {
  const [datasets, setDatasets] = useState<DatasetListItem[]>([])
  const [experiments, setExperiments] = useState<ExperimentListItem[]>([])

  useEffect(() => {
    fetchDatasets().then(setDatasets).catch(() => {})
    fetchExperiments().then(setExperiments).catch(() => {})
  }, [])

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-800">Dashboard</h2>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4">
        <Card>
          <p className="text-3xl font-bold text-blue-600">{datasets.length}</p>
          <p className="text-sm text-gray-500 mt-1">Datasets</p>
        </Card>
        <Card>
          <p className="text-3xl font-bold text-green-600">{experiments.length}</p>
          <p className="text-sm text-gray-500 mt-1">Experiments</p>
        </Card>
        <Card>
          <p className="text-3xl font-bold text-purple-600">
            {experiments.filter(e => e.status === 'completed').length}
          </p>
          <p className="text-sm text-gray-500 mt-1">Completed</p>
        </Card>
      </div>

      {/* Recent experiments */}
      <Card title="Recent Experiments">
        {experiments.length === 0 ? (
          <p className="text-sm text-gray-400">
            No experiments yet.{' '}
            <Link to="/builder" className="text-blue-500 hover:underline">
              Create one
            </Link>
          </p>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-gray-500 border-b">
                <th className="pb-2">Name</th>
                <th className="pb-2">Type</th>
                <th className="pb-2">Status</th>
                <th className="pb-2">Best</th>
                <th className="pb-2">Duration</th>
              </tr>
            </thead>
            <tbody>
              {experiments.slice(0, 5).map(e => (
                <tr key={e.experiment_id} className="border-b border-gray-50">
                  <td className="py-2">
                    <Link to={`/experiments/${e.experiment_id}`} className="text-blue-600 hover:underline">
                      {e.name}
                    </Link>
                  </td>
                  <td className="py-2 text-gray-500">{e.experiment_type}</td>
                  <td className="py-2"><StatusBadge status={e.status} /></td>
                  <td className="py-2 text-gray-600">{e.best_algorithm ?? '-'}</td>
                  <td className="py-2 text-gray-500">{e.duration_seconds ? `${e.duration_seconds.toFixed(1)}s` : '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  )
}
