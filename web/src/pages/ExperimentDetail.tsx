import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import Card from '../components/shared/Card'
import StatusBadge from '../components/shared/StatusBadge'
import { fetchExperiment, type ExperimentDetail as EDetail } from '../api/experiments'

export default function ExperimentDetail() {
  const { id } = useParams<{ id: string }>()
  const [exp, setExp] = useState<EDetail | null>(null)

  useEffect(() => {
    if (id) fetchExperiment(id).then(setExp).catch(() => {})
  }, [id])

  if (!exp) return <p className="text-gray-400">Loading...</p>

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <h2 className="text-2xl font-bold text-gray-800">{exp.name}</h2>
        <StatusBadge status={exp.status} />
      </div>

      {exp.error_message && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3">
          <p className="text-sm text-red-700">{exp.error_message}</p>
        </div>
      )}

      <div className="grid grid-cols-3 gap-4">
        <Card title="Summary">
          <dl className="space-y-2 text-sm">
            <div className="flex justify-between"><dt className="text-gray-500">ID</dt><dd className="font-mono text-xs">{exp.experiment_id}</dd></div>
            <div className="flex justify-between"><dt className="text-gray-500">Type</dt><dd>{exp.experiment_type}</dd></div>
            <div className="flex justify-between"><dt className="text-gray-500">Best</dt><dd className="font-medium">{exp.best_algorithm ?? '-'}</dd></div>
            <div className="flex justify-between"><dt className="text-gray-500">Duration</dt><dd>{exp.duration_seconds ? `${exp.duration_seconds.toFixed(1)}s` : '-'}</dd></div>
            <div className="flex justify-between"><dt className="text-gray-500">Samples</dt><dd>{exp.sample_count ?? '-'}</dd></div>
            <div className="flex justify-between"><dt className="text-gray-500">Features</dt><dd>{exp.feature_count ?? '-'}</dd></div>
          </dl>
        </Card>

        {exp.metrics && (
          <Card title="Metrics">
            <dl className="space-y-2 text-sm">
              {Object.entries(exp.metrics).map(([k, v]) => (
                <div key={k} className="flex justify-between">
                  <dt className="text-gray-500">{k}</dt>
                  <dd className="font-mono">{v.toFixed(4)}</dd>
                </div>
              ))}
            </dl>
          </Card>
        )}

        {exp.all_algorithm_scores && exp.all_algorithm_scores.length > 0 && (
          <Card title="Algorithm Ranking">
            <div className="space-y-2">
              {exp.all_algorithm_scores.map((s: any) => (
                <div key={s.algorithm} className="flex items-center justify-between text-sm">
                  <span className={s.rank === 1 ? 'font-bold text-blue-600' : 'text-gray-600'}>
                    #{s.rank} {s.algorithm}
                  </span>
                  <span className="text-xs text-gray-400 font-mono">
                    {Object.entries(s.metrics).map(([k, v]: any) => `${k}=${v.toFixed(4)}`).join(', ')}
                  </span>
                </div>
              ))}
            </div>
          </Card>
        )}
      </div>

      {exp.feature_importance && exp.feature_importance.length > 0 && (
        <Card title="Feature Importance (Top 15)">
          <div className="space-y-1">
            {exp.feature_importance.slice(0, 15).map((f: any) => (
              <div key={f.feature} className="flex items-center gap-2 text-sm">
                <span className="w-40 truncate font-mono text-xs">{f.feature}</span>
                <div className="flex-1 h-3 bg-gray-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blue-500 rounded-full"
                    style={{ width: `${(f.importance * 100).toFixed(1)}%` }}
                  />
                </div>
                <span className="text-xs text-gray-400 w-14 text-right">{(f.importance * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}
