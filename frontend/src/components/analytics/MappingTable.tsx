import React from 'react'
import { useWorkbenchStore } from '../../store/workbenchStore'
import { api } from '../../services/api'
import { Network, Play, RefreshCw, CheckCircle, Cpu } from 'lucide-react'

export const MappingTable: React.FC = () => {
  const {
    sessionId,
    mapping,
    setMapping,
    isMappingRunning,
    setIsMappingRunning
  } = useWorkbenchStore()

  const handleRunMapping = async () => {
    if (!sessionId) return
    setIsMappingRunning(true)
    try {
      const res = await api.runMapping(sessionId)
      setMapping(res)
    } catch (err) {
      console.error("Mapping failed:", err)
    } finally {
      setIsMappingRunning(false)
    }
  }

  return (
    <div className="p-3 h-full flex flex-col justify-between select-none overflow-y-auto">
      <div className="space-y-3">
        {/* Header with Run Button */}
        <div className="flex items-center justify-between">
          <div>
            <h4 className="text-xs font-bold text-slate-200">
              Automated Physics Node → Neural Layer Mapping
            </h4>
            <p className="text-[11px] text-slate-400">
              Discovers which neural weights encode specific physical parameters via perturbation response.
            </p>
          </div>

          <button
            onClick={handleRunMapping}
            disabled={isMappingRunning}
            className="flex items-center gap-1.5 px-3 py-1 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white text-xs font-semibold rounded-lg shadow transition-all cursor-pointer disabled:opacity-50"
          >
            {isMappingRunning ? (
              <>
                <RefreshCw className="w-3 h-3 animate-spin" />
                <span>Running Perturbation Mapping...</span>
              </>
            ) : (
              <>
                <Play className="w-3 h-3 fill-current" />
                <span>Run Systematic Layer Mapping</span>
              </>
            )}
          </button>
        </div>

        {/* Mapping Table */}
        {mapping && mapping.length > 0 ? (
          <div className="border border-slate-800 rounded-lg overflow-hidden bg-slate-950/60">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-900/90 text-slate-400 text-[10px] uppercase tracking-wider border-b border-slate-800">
                <tr>
                  <th className="p-2">Physical Node Concept</th>
                  <th className="p-2">Symbol</th>
                  <th className="p-2">Mapped Neural Layer</th>
                  <th className="p-2">Confidence Score</th>
                  <th className="p-2">Alignment %</th>
                  <th className="p-2">Mapping Strategy</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {mapping.map((m, idx) => (
                  <tr key={idx} className="hover:bg-slate-900/40 transition-colors">
                    <td className="p-2 font-semibold text-slate-200">
                      {m.nodeLabel}
                    </td>
                    <td className="p-2 font-mono font-bold text-cyan-300">
                      {m.nodeSymbol || "x"}
                    </td>
                    <td className="p-2 font-mono text-purple-300 bg-purple-950/20 rounded">
                      {m.mappedLayer}
                    </td>
                    <td className="p-2">
                      <div className="flex items-center gap-2">
                        <div className="w-20 bg-slate-800 h-1.5 rounded-full overflow-hidden">
                          <div
                            className="bg-gradient-to-r from-blue-500 to-cyan-400 h-full rounded-full"
                            style={{ width: `${(m.confidence * 100).toFixed(0)}%` }}
                          />
                        </div>
                        <span className="font-mono text-[11px] text-cyan-400 font-semibold">
                          {(m.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    </td>
                    <td className="p-2 font-mono text-emerald-400 font-semibold">
                      {m.alignmentPct.toFixed(1)}%
                    </td>
                    <td className="p-2">
                      <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-slate-900 text-slate-300 border border-slate-700">
                        {m.strategy}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="p-6 text-center text-slate-500 text-xs border border-dashed border-slate-800 rounded-lg">
            Click "Run Systematic Layer Mapping" to discover neural representations of physical ODE parameters.
          </div>
        )}
      </div>
    </div>
  )
}
