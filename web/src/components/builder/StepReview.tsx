import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useBuilderStore } from '../../store/builder'
import { submitExperiment, validateExperiment, type ExperimentRequest } from '../../api/experiments'
import { stringify } from 'yaml'

export default function StepReview() {
  const { config, prevStep, reset } = useBuilderStore()
  const navigate = useNavigate()
  const [submitting, setSubmitting] = useState(false)
  const [validating, setValidating] = useState(false)
  const [errors, setErrors] = useState<string[]>([])
  const [validated, setValidated] = useState(false)

  // Build the clean config (remove undefined/empty)
  const cleanConfig: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(config)) {
    if (v !== undefined && v !== null && v !== '' && !(Array.isArray(v) && v.length === 0)) {
      cleanConfig[k] = v
    }
  }

  const yamlStr = stringify(cleanConfig)

  const handleValidate = async () => {
    setValidating(true)
    setErrors([])
    try {
      const result = await validateExperiment(config as ExperimentRequest)
      if (result.valid) {
        setValidated(true)
      } else {
        setErrors(result.errors.map(e => `[${e.code}] ${e.message}`))
      }
    } catch (err: any) {
      setErrors([err.response?.data?.detail ?? err.message])
    } finally {
      setValidating(false)
    }
  }

  const handleSubmit = async () => {
    setSubmitting(true)
    setErrors([])
    try {
      const result = await submitExperiment(config as ExperimentRequest)
      reset()
      navigate(`/experiments/${result.experiment_id}`)
    } catch (err: any) {
      setErrors([err.response?.data?.detail ?? err.message])
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">Step 7: Review & Submit</h3>
      <p className="text-sm text-gray-500">Review the generated YAML config and submit your experiment.</p>

      {/* YAML preview */}
      <div className="bg-gray-900 text-green-400 rounded-lg p-4 overflow-x-auto">
        <pre className="text-xs font-mono whitespace-pre">{yamlStr}</pre>
      </div>

      {/* Errors */}
      {errors.length > 0 && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3">
          <p className="text-sm font-medium text-red-800 mb-1">Validation Errors:</p>
          {errors.map((e, i) => (
            <p key={i} className="text-xs text-red-700">{e}</p>
          ))}
        </div>
      )}

      {validated && errors.length === 0 && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-3">
          <p className="text-sm text-green-700">Config validated successfully.</p>
        </div>
      )}

      <div className="flex justify-between pt-4">
        <button onClick={prevStep} className="px-5 py-2 border text-sm rounded-lg hover:bg-gray-50">Back</button>
        <div className="flex gap-2">
          <button
            onClick={handleValidate}
            disabled={validating}
            className="px-5 py-2 border border-blue-600 text-blue-600 text-sm rounded-lg hover:bg-blue-50 disabled:opacity-40"
          >
            {validating ? 'Validating...' : 'Validate'}
          </button>
          <button
            onClick={handleSubmit}
            disabled={submitting}
            className="px-5 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 disabled:opacity-40"
          >
            {submitting ? 'Running...' : 'Submit Experiment'}
          </button>
        </div>
      </div>
    </div>
  )
}
