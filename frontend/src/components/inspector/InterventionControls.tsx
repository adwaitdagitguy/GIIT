import React from 'react'
import { useWorkbenchStore } from '../../store/workbenchStore'
import { api } from '../../services/api'
import { InterventionType } from '../../types/api'
import {
  Sliders,
  Zap,
  Play,
  RotateCw,
  Gauge,
  ShieldAlert,
  Flame,
  LineChart
} from 'lucide-react'

export const InterventionControls: React.FC = () => {
  const {
    sessionId,
    modelMetadata,
    selectedLayer,
    setSelectedLayer,
    interventionType,
    setInterventionType,
    interventionStrength,
    setInterventionStrength,
    setInterventionResult,
    setSweepResult,
    isInterventionRunning,
    setIsInterventionRunning,
    isSweepRunning,
    setIsSweepRunning,
    setActiveAnalyticsTab
  } = useWorkbenchStore()

  const availableLayers = modelMetadata?.perturbableTensors?.length
    ? modelMetadata.perturbableTensors
    : ['fcs.0.weight', 'fch.0.weight', 'fch.1.weight', 'fce.0.weight']

  // Run single intervention
  const handleRunIntervention = async () => {
    if (!sessionId) return
    setIsInterventionRunning(true)

    try {
      const result = await api.runIntervention(
        sessionId,
        selectedLayer,
        interventionType,
        interventionStrength
      )
      setInterventionResult(result)
      setActiveAnalyticsTab('trajectory')
    } catch (err) {
      console.error("Intervention run failed:", err)
    } finally {
      setIsInterventionRunning(false)
    }
  }

  // Run strength sweep
  const handleRunSweep = async () => {
    if (!sessionId) return
    setIsSweepRunning(true)

    try {
      const strengths = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
      const result = await api.runSweep(
        sessionId,
        selectedLayer,
        interventionType,
        strengths
      )
      setSweepResult(result)
      setActiveAnalyticsTab('sweep')
    } catch (err) {
      console.error("Sweep run failed:", err)
    } finally {
      setIsSweepRunning(false)
    }
  }

  return (
    <div className="p-3.5 space-y-3.5 bg-slate-900/40 select-none">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Zap className="w-4 h-4 text-amber-400" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
            Intervention Studio
          </h3>
        </div>
        <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-300 border border-amber-500/20">
          Perturbation
        </span>
      </div>

      {/* Target Neural Layer */}
      <div className="space-y-1">
        <label className="text-[11px] font-semibold text-slate-300 flex items-center justify-between">
          <span>Target PINN Layer:</span>
          <span className="text-[10px] font-mono text-cyan-400 font-semibold">{selectedLayer}</span>
        </label>
        <select
          value={selectedLayer}
          onChange={(e) => setSelectedLayer(e.target.value)}
          className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 font-mono"
        >
          {availableLayers.map((layer) => (
            <option key={layer} value={layer}>
              {layer}
            </option>
          ))}
        </select>
      </div>

      {/* Intervention Type */}
      <div className="space-y-1">
        <label className="text-[11px] font-semibold text-slate-300">
          Intervention Method:
        </label>
        <div className="grid grid-cols-2 gap-1.5">
          {[
            { id: 'noise', label: 'Noise Injection', desc: 'Gaussian jitter' },
            { id: 'masking', label: 'Weight Masking', desc: 'Ablate/Zero' },
            { id: 'gain', label: 'Gain Scaling', desc: 'Amplify/Attenuate' },
            { id: 'clamp', label: 'Value Clamp', desc: 'Bound activations' }
          ].map((type) => (
            <button
              key={type.id}
              onClick={() => setInterventionType(type.id as InterventionType)}
              className={`p-2 rounded-lg border text-left transition-all cursor-pointer ${
                interventionType === type.id
                  ? 'bg-amber-950/40 border-amber-500/60 text-amber-200 ring-1 ring-amber-500/30'
                  : 'bg-slate-950/60 hover:bg-slate-800/40 border-slate-800 text-slate-400'
              }`}
            >
              <div className="text-xs font-semibold">{type.label}</div>
              <div className="text-[9px] text-slate-500 truncate">{type.desc}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Intervention Strength Slider */}
      <div className="space-y-1.5 pt-1">
        <div className="flex items-center justify-between text-[11px] font-semibold">
          <span className="text-slate-300">Perturbation Strength (α):</span>
          <span className="font-mono text-cyan-400 bg-cyan-950/60 px-2 py-0.5 rounded border border-cyan-500/30">
            {(interventionStrength * 100).toFixed(0)}% ({interventionStrength.toFixed(2)})
          </span>
        </div>

        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={interventionStrength}
          onChange={(e) => setInterventionStrength(parseFloat(e.target.value))}
          className="w-full accent-cyan-400 bg-slate-800 rounded-lg h-1.5 cursor-pointer"
        />

        {/* Quick Presets */}
        <div className="flex justify-between gap-1 pt-0.5">
          {[0.1, 0.25, 0.5, 0.75, 1.0].map((preset) => (
            <button
              key={preset}
              onClick={() => setInterventionStrength(preset)}
              className={`px-2 py-0.5 rounded text-[10px] font-mono transition-colors cursor-pointer ${
                interventionStrength === preset
                  ? 'bg-cyan-500 text-slate-950 font-bold'
                  : 'bg-slate-950/80 hover:bg-slate-800 text-slate-400 border border-slate-800'
              }`}
            >
              {(preset * 100).toFixed(0)}%
            </button>
          ))}
        </div>
      </div>

      {/* Action Buttons */}
      <div className="pt-2 space-y-2">
        <button
          onClick={handleRunIntervention}
          disabled={isInterventionRunning}
          className="w-full py-2.5 px-3 bg-gradient-to-r from-amber-600 via-orange-600 to-rose-600 hover:from-amber-500 hover:to-rose-500 text-white text-xs font-semibold rounded-lg shadow-lg shadow-amber-500/20 transition-all flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50"
        >
          {isInterventionRunning ? (
            <>
              <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              <span>Simulating Perturbation...</span>
            </>
          ) : (
            <>
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>Apply Layer Intervention</span>
            </>
          )}
        </button>

        <button
          onClick={handleRunSweep}
          disabled={isSweepRunning}
          className="w-full py-2 px-3 bg-slate-800/80 hover:bg-slate-700/80 border border-slate-700 text-slate-200 text-xs font-medium rounded-lg transition-all flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50"
        >
          {isSweepRunning ? (
            <>
              <div className="w-3.5 h-3.5 border-2 border-cyan-400/30 border-t-cyan-400 rounded-full animate-spin" />
              <span>Sweeping (0.0 → 1.0)...</span>
            </>
          ) : (
            <>
              <LineChart className="w-3.5 h-3.5 text-cyan-400" />
              <span>Run Parameter Sweep Loop</span>
            </>
          )}
        </button>
      </div>
    </div>
  )
}
