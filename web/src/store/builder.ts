import { create } from 'zustand'
import type { ExperimentRequest } from '../api/experiments'

export interface BuilderState {
  step: number
  config: Partial<ExperimentRequest>
  setStep: (step: number) => void
  nextStep: () => void
  prevStep: () => void
  updateConfig: (patch: Partial<ExperimentRequest>) => void
  reset: () => void
}

const INITIAL: Partial<ExperimentRequest> = {}

export const useBuilderStore = create<BuilderState>((set) => ({
  step: 0,
  config: { ...INITIAL },
  setStep: (step) => set({ step }),
  nextStep: () => set((s) => ({ step: Math.min(s.step + 1, 6) })),
  prevStep: () => set((s) => ({ step: Math.max(s.step - 1, 0) })),
  updateConfig: (patch) =>
    set((s) => ({ config: { ...s.config, ...patch } })),
  reset: () => set({ step: 0, config: { ...INITIAL } }),
}))
