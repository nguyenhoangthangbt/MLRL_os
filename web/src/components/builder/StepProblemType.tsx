import { useBuilderStore } from '../../store/builder'

const TYPES = [
  {
    value: 'time_series',
    label: 'Time-Series Forecasting',
    desc: 'Predict future values from periodic system snapshots (Layer 3).',
  },
  {
    value: 'entity_classification',
    label: 'Entity Classification',
    desc: 'Classify entities based on trajectory observations (Layer 2).',
  },
]

export default function StepProblemType() {
  const { config, updateConfig, prevStep, nextStep } = useBuilderStore()

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">Step 3: Problem Type</h3>
      <p className="text-sm text-gray-500">Select the type of prediction problem.</p>

      <div className="space-y-3">
        {TYPES.map(t => (
          <label
            key={t.value}
            className={`block p-4 rounded-lg border cursor-pointer transition ${
              config.experiment_type === t.value
                ? 'border-blue-500 bg-blue-50'
                : 'border-gray-200 hover:border-gray-300'
            }`}
          >
            <div className="flex items-center gap-3">
              <input
                type="radio"
                name="problem_type"
                checked={config.experiment_type === t.value}
                onChange={() => updateConfig({ experiment_type: t.value })}
                className="accent-blue-600"
              />
              <div>
                <p className="font-medium text-sm">{t.label}</p>
                <p className="text-xs text-gray-400 mt-0.5">{t.desc}</p>
              </div>
            </div>
          </label>
        ))}
      </div>

      <div className="flex justify-between pt-4">
        <button onClick={prevStep} className="px-5 py-2 border text-sm rounded-lg hover:bg-gray-50">Back</button>
        <button
          onClick={nextStep}
          disabled={!config.experiment_type}
          className="px-5 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 disabled:opacity-40"
        >
          Next
        </button>
      </div>
    </div>
  )
}
