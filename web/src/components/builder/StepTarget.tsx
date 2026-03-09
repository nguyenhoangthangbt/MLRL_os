import { useEffect, useState } from 'react'
import { useBuilderStore } from '../../store/builder'
import { fetchTargets, type TargetInfo } from '../../api/datasets'

export default function StepTarget() {
  const { config, updateConfig, prevStep, nextStep } = useBuilderStore()
  const [targets, setTargets] = useState<TargetInfo[]>([])

  useEffect(() => {
    if (!config.dataset_id) return
    fetchTargets(config.dataset_id).then(data => {
      const list =
        config.experiment_type === 'time_series'
          ? data.time_series_targets
          : data.entity_targets
      setTargets(list)
      // Auto-select the default target
      const def = list.find(t => t.is_default)
      if (def && !config.target) updateConfig({ target: def.column })
    }).catch(() => {})
  }, [config.dataset_id, config.experiment_type])

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">Step 4: Select Target</h3>
      <p className="text-sm text-gray-500">Choose the column to predict.</p>

      {targets.length === 0 ? (
        <p className="text-sm text-gray-400">No targets discovered for this problem type.</p>
      ) : (
        <div className="space-y-2">
          {targets.map(t => (
            <label
              key={t.column}
              className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition ${
                config.target === t.column
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <input
                type="radio"
                name="target"
                checked={config.target === t.column}
                onChange={() => updateConfig({ target: t.column })}
                className="accent-blue-600"
              />
              <div>
                <p className="font-medium text-sm">
                  {t.column}
                  {t.is_default && <span className="ml-2 text-xs text-blue-500">(default)</span>}
                </p>
                <p className="text-xs text-gray-400">
                  {t.task_type}
                  {t.classes && ` — ${t.classes.length} classes`}
                  {t.unique_count != null && ` — ${t.unique_count} unique`}
                </p>
              </div>
            </label>
          ))}
        </div>
      )}

      <div className="flex justify-between pt-4">
        <button onClick={prevStep} className="px-5 py-2 border text-sm rounded-lg hover:bg-gray-50">Back</button>
        <button
          onClick={nextStep}
          disabled={!config.target}
          className="px-5 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 disabled:opacity-40"
        >
          Next
        </button>
      </div>
    </div>
  )
}
