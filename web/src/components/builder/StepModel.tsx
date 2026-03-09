import { useBuilderStore } from '../../store/builder'

const TS_ALGOS = ['lightgbm', 'xgboost', 'random_forest', 'extra_trees', 'linear']
const ENTITY_ALGOS = ['lightgbm', 'xgboost', 'random_forest', 'extra_trees', 'linear']

const TS_METRICS = ['rmse', 'mae', 'mape', 'r2']
const ENTITY_METRICS = ['f1_weighted', 'f1_macro', 'auc_roc', 'precision', 'recall', 'accuracy']

export default function StepModel() {
  const { config, updateConfig, prevStep, nextStep } = useBuilderStore()
  const isTS = config.experiment_type === 'time_series'
  const algos = isTS ? TS_ALGOS : ENTITY_ALGOS
  const metrics = isTS ? TS_METRICS : ENTITY_METRICS
  const selected = config.algorithms ?? []
  const selectedMetrics = config.metrics ?? []

  const toggleAlgo = (a: string) => {
    const next = selected.includes(a) ? selected.filter(x => x !== a) : [...selected, a]
    updateConfig({ algorithms: next })
  }

  const toggleMetric = (m: string) => {
    const next = selectedMetrics.includes(m) ? selectedMetrics.filter(x => x !== m) : [...selectedMetrics, m]
    updateConfig({ metrics: next })
  }

  return (
    <div className="space-y-5">
      <h3 className="text-lg font-semibold">Step 6: Model Configuration</h3>

      {/* Algorithms */}
      <div>
        <label className="block text-xs font-medium text-gray-600 mb-2">Algorithms</label>
        <div className="flex flex-wrap gap-2">
          {algos.map(a => (
            <button
              key={a}
              onClick={() => toggleAlgo(a)}
              className={`px-3 py-1.5 text-xs rounded-lg border transition ${
                selected.includes(a)
                  ? 'bg-blue-600 text-white border-blue-600'
                  : 'bg-white text-gray-600 border-gray-200 hover:border-gray-400'
              }`}
            >
              {a}
            </button>
          ))}
        </div>
      </div>

      {/* Metrics */}
      <div>
        <label className="block text-xs font-medium text-gray-600 mb-2">Metrics</label>
        <div className="flex flex-wrap gap-2">
          {metrics.map(m => (
            <button
              key={m}
              onClick={() => toggleMetric(m)}
              className={`px-3 py-1.5 text-xs rounded-lg border transition ${
                selectedMetrics.includes(m)
                  ? 'bg-green-600 text-white border-green-600'
                  : 'bg-white text-gray-600 border-gray-200 hover:border-gray-400'
              }`}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      {/* CV folds */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-xs font-medium text-gray-600 mb-1">CV Folds</label>
          <input
            type="number"
            min={2}
            max={20}
            value={config.cv_folds ?? 5}
            onChange={e => updateConfig({ cv_folds: parseInt(e.target.value) || 5 })}
            className="w-full border rounded-lg px-3 py-2 text-sm"
          />
        </div>
        <div>
          <label className="block text-xs font-medium text-gray-600 mb-1">Seed</label>
          <input
            type="number"
            min={0}
            value={config.seed ?? 42}
            onChange={e => updateConfig({ seed: parseInt(e.target.value) || 42 })}
            className="w-full border rounded-lg px-3 py-2 text-sm"
          />
        </div>
      </div>

      {/* Toggles */}
      <div className="flex gap-6">
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={config.hyperparameter_tuning ?? false}
            onChange={e => updateConfig({ hyperparameter_tuning: e.target.checked })}
            className="accent-blue-600"
          />
          Hyperparameter Tuning (Optuna)
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={config.handle_imbalance ?? false}
            onChange={e => updateConfig({ handle_imbalance: e.target.checked })}
            className="accent-blue-600"
          />
          Handle Imbalance
        </label>
      </div>

      <div className="flex justify-between pt-4">
        <button onClick={prevStep} className="px-5 py-2 border text-sm rounded-lg hover:bg-gray-50">Back</button>
        <button onClick={nextStep} className="px-5 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700">Next</button>
      </div>
    </div>
  )
}
