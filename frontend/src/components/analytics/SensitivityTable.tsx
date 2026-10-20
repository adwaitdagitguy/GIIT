import React from 'react'
import { useWorkbenchStore } from '../../store/workbenchStore'
import { api } from '../../services/api'
import { CheckCircle2, AlertTriangle, Play, RefreshCw, Gauge } from 'lucide-react'

export const SensitivityTable: React.FC = () => {
  const {
    sessionId,
    fullAnalysis,
    setFullAnalysis,
    isAnalyzing,
    setIsAnalyzing
  } = useWorkbenchStore()

  const handleRunAnalysis = async () => {
    if (!sessionId) return
    setIsAnalyzing(true)
    try {
      const res = await api.runFullAnalysis(sessionId)
      setFullAnalysis(res)
    } catch (err) {
      console.error("Full analysis failed:", err)
    } finally {
      setIsAnalyzing(false)
    }
  }

  return (
    <div className="p-3 h-full flex flex-col justify-between select-none overflow-y-auto">
      <div className="space-y-3">
        {/* Top Summary Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-slate-950/80 px-2.5 py-1 rounded-lg border border-slate-800 text-xs">
              <span className="text-slate-400">Physics Residual Norm: </span>
              <span className="font-mono font-bold text-rose-400">
                {fullAnalysis ? fullAnalysis.physicsResidualNorm.toFixed(5) : "—"}
              </span>
            </div>
            <div className="bg-slate-950/80 px-2.5 py-1 rounded-lg border border-slate-800 text-xs">
              <span className="text-slate-400">Baseline L2 Norm: </span>
              <span className="font-mono font-bold text-cyan-400">
                {fullAnalysis ? fullAnalysis.l2NormBaseline.toFixed(4) : "—"}
              </span>
            </div>
          </div>

          <button
            onClick={handleRunAnalysis}
            disabled={isAnalyzing}
            className="flex items-center gap-1.5 px-3 py-1 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white text-xs font-semibold rounded-lg shadow transition-all cursor-pointer disabled:opacity-50"
          >
            {isAnalyzing ? (
              <>
                <RefreshCw className="w-3 h-3 animate-spin" />
                <span>Computing Gradients...</span>
              </>
            ) : (
              <>
                <Play className="w-3 h-3 fill-current" />
                <span>Re-Run Sensitivity Analysis</span>
              </>
            )}
          </button>
        </div>

        {/* Sensitivity Table */}
        {fullAnalysis && fullAnalysis.sensitivity.length > 0 ? (
          <div className="border border-slate-800 rounded-lg overflow-hidden bg-slate-950/60">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-900/90 text-slate-400 text-[10px] uppercase tracking-wider border-b border-slate-800">
                <tr>
                  <th className="p-2">Physical Parameter</th>
                  <th className="p-2">Autograd Sign</th>
                  <th className="p-2">Expected Physical Sign</th>
                  <th className="p-2">Consistency Status</th>
                  <th className="p-2">Gradient Magnitude</th>
                  <th className="p-2">Physical Context</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {fullAnalysis.sensitivity.map((item, idx) => (
                  <tr key={idx} className="hover:bg-slate-900/40 transition-colors">
                    <td className="p-2 font-mono font-semibold text-slate-200">
                      {item.label} ({item.parameter})
                    </td>
                    <td className="p-2 font-mono font-bold text-cyan-300">
                      [{item.gradSign}]
                    </td>
                    <td className="p-2 font-mono font-bold text-amber-300">
                      [{item.expectedSign}]
                    </td>
                    <td className="p-2">
                      {item.consistent ? (
                        <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-emerald-950/60 text-emerald-400 border border-emerald-500/30 text-[10px] font-semibold">
                          <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                          Physically Consistent
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-rose-950/60 text-rose-400 border border-rose-500/30 text-[10px] font-semibold">
                          <AlertTriangle className="w-3 h-3 text-rose-400" />
                          Sign Violation
                        </span>
                      )}
                    </td>
                    <td className="p-2 font-mono text-slate-300">
                      {item.magnitude.toFixed(4)}
                    </td>
                    <td className="p-2 text-slate-400 text-[11px] truncate max-w-xs">
                      {item.description || "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="p-6 text-center text-slate-500 text-xs border border-dashed border-slate-800 rounded-lg">
            Click "Re-Run Sensitivity Analysis" to evaluate PINN gradient directions against ODE parameters.
          </div>
        )}
      </div>
    </div>
  )
}
