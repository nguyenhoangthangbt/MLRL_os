import { useBuilderStore } from '../../store/builder'

export default function StepFeatures() {
  const { config, updateConfig, prevStep, nextStep } = useBuilderStore()
  const isTS = config.experiment_type === 'time_series'

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">Step 5: Feature Configuration</h3>
      <p className="text-sm text-gray-500">
        {isTS
          ? 'Configure time-series feature engineering parameters.'
          : 'Configure entity feature engineering parameters.'}
      </p>

      {isTS ? (
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1">Lookback</label>
            <input
              type="text"
              value={config.lookback ?? ''}
              onChange={e => updateConfig({ lookback: e.target.value })}
              placeholder="e.g. 8h"
              className="w-full border rounded-lg px-3 py-2 text-sm"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1">Horizon</label>
            <input
              type="text"
              value={config.horizon ?? ''}
              onChange={e => updateConfig({ horizon: e.target.value })}
              placeholder="e.g. 1h"
              className="w-full border rounded-lg px-3 py-2 text-sm"
            />
          </div>
        </div>
      ) : (
        <div>
          <label className="block text-xs font-medium text-gray-600 mb-1">Observation Point</label>
          <select
            value={config.observation_point ?? 'all_steps'}
            onChange={e => updateConfig({ observation_point: e.target.value })}
            className="w-full border rounded-lg px-3 py-2 text-sm"
          >
            <option value="all_steps">All Steps</option>
            <option value="last_step">Last Step</option>
            <option value="mid_point">Mid Point</option>
          </select>
        </div>
      )}

      <p className="text-xs text-gray-400">
        Leave blank for sensible defaults. Advanced options can be set in the YAML config.
      </p>

      <div className="flex justify-between pt-4">
        <button onClick={prevStep} className="px-5 py-2 border text-sm rounded-lg hover:bg-gray-50">Back</button>
        <button onClick={nextStep} className="px-5 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700">Next</button>
      </div>
    </div>
  )
}
