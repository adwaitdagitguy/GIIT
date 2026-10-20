import React, { useState } from 'react'
import { useWorkbenchStore } from '../../store/workbenchStore'
import { Sparkles, ArrowRight, ArrowLeft, X, Check } from 'lucide-react'

interface TourStep {
  title: string
  description: string
  target: string
  placement: 'right' | 'left' | 'top' | 'bottom'
}

const tourSteps: TourStep[] = [
  {
    title: "1. Upload PINN Model & Ruleset",
    description: "Upload your PyTorch model (.py), trained weights (.pt), and physics ODE causal graph (.json) in the left sidebar, or click 'Load Demo Model' to get started immediately.",
    target: "sidebar",
    placement: "right"
  },
  {
    title: "2. Causal Physics Graph Canvas",
    description: "Explore the interactive causal graph showing physical inputs, latent representation layers, parameters (μ, k), derivatives (ẏ, ÿ), and the governing ODE loss residual.",
    target: "canvas",
    placement: "bottom"
  },
  {
    title: "3. Node Inspector",
    description: "Click any physical node in the graph to inspect its mathematical formula, physical description, default parameters, and qualitative trend expectations.",
    target: "inspector",
    placement: "left"
  },
  {
    title: "4. Intervention Studio",
    description: "Select a neural layer, choose an intervention method (Noise injection, weight masking, gain scaling), set the strength slider, and click 'Apply Layer Intervention'.",
    target: "controls",
    placement: "left"
  },
  {
    title: "5. Analytics & Physics Diagnostics",
    description: "Inspect the baseline vs perturbed trajectory divergence, compute autograd sensitivity gradients against ODE parameters, and verify neural layer mappings.",
    target: "analytics",
    placement: "top"
  }
]

export const GuidedTour: React.FC = () => {
  const { isTourActive, setIsTourActive } = useWorkbenchStore()
  const [currentStep, setCurrentStep] = useState(0)

  if (!isTourActive) return null

  const step = tourSteps[currentStep]

  const handleNext = () => {
    if (currentStep < tourSteps.length - 1) {
      setCurrentStep(currentStep + 1)
    } else {
      setIsTourActive(false)
      setCurrentStep(0)
    }
  }

  const handlePrev = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-xs z-50 flex items-center justify-center p-4 select-none">
      <div className="bg-slate-900 border border-cyan-500/40 rounded-2xl max-w-md w-full p-6 shadow-2xl shadow-cyan-500/10 space-y-4 animate-in fade-in zoom-in duration-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 flex items-center justify-center">
              <Sparkles className="w-4 h-4" />
            </div>
            <span className="text-xs font-bold text-cyan-400 uppercase tracking-wider">
              Guided Tour • Step {currentStep + 1} of {tourSteps.length}
            </span>
          </div>
          <button
            onClick={() => setIsTourActive(false)}
            className="text-slate-400 hover:text-white p-1"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div>
          <h3 className="text-base font-bold text-slate-100 mb-2">{step.title}</h3>
          <p className="text-xs text-slate-300 leading-relaxed">{step.description}</p>
        </div>

        {/* Progress Bar */}
        <div className="flex gap-1.5 pt-1">
          {tourSteps.map((_, i) => (
            <div
              key={i}
              className={`h-1 flex-1 rounded-full transition-colors ${
                i <= currentStep ? 'bg-cyan-400' : 'bg-slate-800'
              }`}
            />
          ))}
        </div>

        <div className="flex items-center justify-between pt-3 border-t border-slate-800">
          <button
            onClick={handlePrev}
            disabled={currentStep === 0}
            className="flex items-center gap-1 text-xs text-slate-400 hover:text-white disabled:opacity-30 cursor-pointer disabled:cursor-not-allowed"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Previous</span>
          </button>

          <button
            onClick={handleNext}
            className="flex items-center gap-1.5 px-4 py-1.5 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white text-xs font-semibold rounded-lg shadow-md shadow-cyan-500/20 transition-all cursor-pointer"
          >
            <span>{currentStep === tourSteps.length - 1 ? 'Finish Tour' : 'Next'}</span>
            {currentStep === tourSteps.length - 1 ? (
              <Check className="w-3.5 h-3.5" />
            ) : (
              <ArrowRight className="w-3.5 h-3.5" />
            )}
          </button>
        </div>
      </div>
    </div>
  )
}
