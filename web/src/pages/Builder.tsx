import { useBuilderStore } from '../store/builder'
import StepDataset from '../components/builder/StepDataset'
import StepExplore from '../components/builder/StepExplore'
import StepProblemType from '../components/builder/StepProblemType'
import StepTarget from '../components/builder/StepTarget'
import StepFeatures from '../components/builder/StepFeatures'
import StepModel from '../components/builder/StepModel'
import StepReview from '../components/builder/StepReview'

const STEPS = [
  { label: 'Dataset', component: StepDataset },
  { label: 'Explore', component: StepExplore },
  { label: 'Problem Type', component: StepProblemType },
  { label: 'Target', component: StepTarget },
  { label: 'Features', component: StepFeatures },
  { label: 'Model', component: StepModel },
  { label: 'Review', component: StepReview },
]

export default function Builder() {
  const { step, setStep } = useBuilderStore()
  const StepComponent = STEPS[step].component

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-800">Experiment Builder</h2>

      {/* Step indicator */}
      <div className="flex gap-1">
        {STEPS.map((s, i) => (
          <button
            key={i}
            onClick={() => setStep(i)}
            className={`flex-1 py-2 text-xs font-medium rounded-t transition-colors ${
              i === step
                ? 'bg-blue-600 text-white'
                : i < step
                ? 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                : 'bg-gray-100 text-gray-400'
            }`}
          >
            {i + 1}. {s.label}
          </button>
        ))}
      </div>

      {/* Step content */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <StepComponent />
      </div>
    </div>
  )
}
