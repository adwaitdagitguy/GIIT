import React from 'react'
import { useWorkbenchStore } from '../../store/workbenchStore'
import {
  Info,
  Layers,
  HelpCircle,
  TrendingUp,
  Activity,
  Sliders,
  CheckCircle2,
  Atom
} from 'lucide-react'

export const NodeInspector: React.FC = () => {
  const { graph, selectedNodeId } = useWorkbenchStore()

  if (!graph) return null

  const selectedNode = graph.nodes.find((n) => n.id === selectedNodeId)

  return (
    <div className="p-3.5 space-y-3 bg-slate-900/40 border-b border-slate-800 select-none">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Atom className="w-4 h-4 text-cyan-400" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
            Node Inspector
          </h3>
        </div>
        {selectedNode && (
          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-cyan-300 border border-slate-700">
            {selectedNode.id}
          </span>
        )}
      </div>

      {selectedNode ? (
        <div className="space-y-2.5">
          {/* Node Title & Symbol */}
          <div className="flex items-start justify-between gap-2 p-2 rounded-lg bg-slate-950/70 border border-slate-800">
            <div>
              <div className="text-xs font-bold text-slate-100">{selectedNode.label}</div>
              <div className="text-[10px] text-slate-400 capitalize">
                Type: <span className="text-cyan-400 font-semibold">{selectedNode.type.replace('_', ' ')}</span>
              </div>
            </div>
            <div className="text-sm font-mono font-bold px-2 py-0.5 rounded bg-slate-900 text-cyan-300 border border-slate-700 shadow-sm">
              {selectedNode.symbol}
            </div>
          </div>

          {/* Mathematical Formula */}
          {selectedNode.formula && (
            <div className="space-y-1">
              <span className="text-[10px] font-semibold text-slate-400">Mathematical Formula:</span>
              <div className="p-2 rounded bg-slate-950 text-xs font-mono text-cyan-300 border border-slate-800 overflow-x-auto">
                {selectedNode.formula}
              </div>
            </div>
          )}

          {/* Description */}
          {selectedNode.description && (
            <div className="space-y-1">
              <span className="text-[10px] font-semibold text-slate-400">Physical Meaning:</span>
              <p className="text-[11px] text-slate-300 bg-slate-950/50 p-2 rounded border border-slate-800/80 leading-relaxed">
                {selectedNode.description}
              </p>
            </div>
          )}

          {/* Parameter Details (if parameter node) */}
          {(selectedNode.default_value !== undefined || selectedNode.expected_sensitivity_sign) && (
            <div className="grid grid-cols-2 gap-1.5 text-[10px]">
              {selectedNode.default_value !== undefined && (
                <div className="bg-slate-950/60 p-1.5 rounded border border-slate-800">
                  <span className="text-slate-400 block">Default Value:</span>
                  <span className="font-bold text-slate-100 font-mono">
                    {selectedNode.default_value} {selectedNode.unit}
                  </span>
                </div>
              )}
              {selectedNode.expected_sensitivity_sign && (
                <div className="bg-slate-950/60 p-1.5 rounded border border-slate-800">
                  <span className="text-slate-400 block">Expected Sign:</span>
                  <span className="font-bold text-amber-300 font-mono">
                    Sign ({selectedNode.expected_sensitivity_sign})
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Qualitative Trend */}
          {selectedNode.trend && (
            <div className="p-2 rounded bg-cyan-950/20 border border-cyan-500/20 text-[11px] text-cyan-200/90 leading-relaxed flex items-start gap-1.5">
              <TrendingUp className="w-3.5 h-3.5 text-cyan-400 shrink-0 mt-0.5" />
              <span>{selectedNode.trend}</span>
            </div>
          )}
        </div>
      ) : (
        <div className="p-4 rounded-lg bg-slate-950/40 border border-dashed border-slate-800 text-center text-slate-500 space-y-1">
          <Info className="w-4 h-4 mx-auto text-slate-600 mb-1" />
          <p className="text-xs text-slate-400">Click any node in the graph</p>
          <p className="text-[10px] text-slate-600">Inspect physical formulas and ODE relations</p>
        </div>
      )}
    </div>
  )
}
