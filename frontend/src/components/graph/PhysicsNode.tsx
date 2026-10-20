import React, { memo } from 'react'
import { Handle, Position, NodeProps } from '@xyflow/react'
import { NodeType } from '../../types/api'

interface PhysicsNodeData {
  label: string
  symbol: string
  type: NodeType
  formula?: string
  unit?: string
  description?: string
  defaultValue?: number
}

const typeStyles: Record<
  NodeType,
  { bg: string; border: string; glow: string; badge: string; text: string; dot: string }
> = {
  input: {
    bg: 'bg-blue-950/70',
    border: 'border-blue-500/60',
    glow: 'shadow-blue-500/20',
    badge: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
    text: 'text-blue-200',
    dot: 'bg-blue-400'
  },
  latent: {
    bg: 'bg-purple-950/70',
    border: 'border-purple-500/60',
    glow: 'shadow-purple-500/20',
    badge: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
    text: 'text-purple-200',
    dot: 'bg-purple-400'
  },
  parameter: {
    bg: 'bg-cyan-950/70',
    border: 'border-cyan-500/60',
    glow: 'shadow-cyan-500/20',
    badge: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30',
    text: 'text-cyan-200',
    dot: 'bg-cyan-400'
  },
  intermediate: {
    bg: 'bg-amber-950/70',
    border: 'border-amber-500/60',
    glow: 'shadow-amber-500/20',
    badge: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
    text: 'text-amber-200',
    dot: 'bg-amber-400'
  },
  physics_loss: {
    bg: 'bg-rose-950/80',
    border: 'border-rose-500/70',
    glow: 'shadow-rose-500/30',
    badge: 'bg-rose-500/20 text-rose-300 border-rose-500/30',
    text: 'text-rose-200',
    dot: 'bg-rose-400'
  },
  output: {
    bg: 'bg-emerald-950/70',
    border: 'border-emerald-500/60',
    glow: 'shadow-emerald-500/20',
    badge: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
    text: 'text-emerald-200',
    dot: 'bg-emerald-400'
  },
  state: {
    bg: 'bg-indigo-950/70',
    border: 'border-indigo-500/60',
    glow: 'shadow-indigo-500/20',
    badge: 'bg-indigo-500/20 text-indigo-300 border-indigo-500/30',
    text: 'text-indigo-200',
    dot: 'bg-indigo-400'
  }
}

export const PhysicsNode = memo(({ data, selected }: NodeProps<any>) => {
  const nodeData = data as PhysicsNodeData
  const nodeType = nodeData.type || 'intermediate'
  const style = typeStyles[nodeType] || typeStyles.intermediate

  return (
    <div
      className={`relative px-3 py-2.5 rounded-xl border backdrop-blur-md transition-all select-none min-w-[170px] max-w-[220px] shadow-lg ${
        style.bg
      } ${style.border} ${style.glow} ${
        selected ? 'ring-2 ring-cyan-400 ring-offset-2 ring-offset-slate-950 scale-105 shadow-xl shadow-cyan-500/30' : 'hover:scale-[1.02]'
      }`}
    >
      {/* Incoming Connection Handle */}
      <Handle
        type="target"
        position={Position.Top}
        className="!w-2.5 !h-2.5 !bg-slate-400 !border-2 !border-slate-950"
      />

      {/* Header: Type Badge & Symbol */}
      <div className="flex items-center justify-between gap-1.5 mb-1.5">
        <span
          className={`text-[9px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded border ${style.badge}`}
        >
          {nodeType.replace('_', ' ')}
        </span>
        <span className="text-xs font-mono font-bold px-1.5 py-0.5 rounded bg-slate-900/90 text-cyan-300 border border-slate-700/80 shadow-inner">
          {nodeData.symbol || 'x'}
        </span>
      </div>

      {/* Node Title & Unit */}
      <div className="text-xs font-semibold text-slate-100 truncate">
        {nodeData.label}
      </div>

      {/* Formula or Value Pill */}
      {nodeData.formula ? (
        <div className="mt-1 text-[10px] font-mono text-slate-300 bg-slate-950/70 px-1.5 py-0.5 rounded border border-slate-800/80 truncate">
          {nodeData.formula}
        </div>
      ) : nodeData.defaultValue !== undefined ? (
        <div className="mt-1 text-[10px] font-mono text-slate-300 bg-slate-950/70 px-1.5 py-0.5 rounded border border-slate-800/80">
          Val: <span className="text-cyan-400 font-bold">{nodeData.defaultValue}</span> {nodeData.unit}
        </div>
      ) : null}

      {/* Outgoing Connection Handle */}
      <Handle
        type="source"
        position={Position.Bottom}
        className="!w-2.5 !h-2.5 !bg-cyan-400 !border-2 !border-slate-950"
      />
    </div>
  )
})
