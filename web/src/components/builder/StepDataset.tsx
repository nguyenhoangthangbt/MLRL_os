import { useEffect, useState } from 'react'
import { useBuilderStore } from '../../store/builder'
import { fetchDatasets, type DatasetListItem } from '../../api/datasets'

export default function StepDataset() {
  const { config, updateConfig, nextStep } = useBuilderStore()
  const [datasets, setDatasets] = useState<DatasetListItem[]>([])

  useEffect(() => { fetchDatasets().then(setDatasets).catch(() => {}) }, [])

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">Step 1: Select Dataset</h3>
      <p className="text-sm text-gray-500">Choose a registered dataset for your experiment.</p>

      {datasets.length === 0 ? (
        <p className="text-sm text-gray-400">No datasets available. Upload one from the Datasets page.</p>
      ) : (
        <div className="space-y-2">
          {datasets.map(d => (
            <label
              key={d.id}
              className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition ${
                config.dataset_id === d.id ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <input
                type="radio"
                name="dataset"
                checked={config.dataset_id === d.id}
                onChange={() => updateConfig({ dataset_id: d.id })}
                className="accent-blue-600"
              />
              <div>
                <p className="font-medium text-sm">{d.name}</p>
                <p className="text-xs text-gray-400">
                  {d.source_type} — {d.snapshot_row_count ?? 0} snapshots, {d.trajectory_row_count ?? 0} trajectories
                </p>
              </div>
            </label>
          ))}
        </div>
      )}

      <div className="flex justify-end pt-4">
        <button
          onClick={nextStep}
          disabled={!config.dataset_id}
          className="px-5 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 disabled:opacity-40"
        >
          Next
        </button>
      </div>
    </div>
  )
}
