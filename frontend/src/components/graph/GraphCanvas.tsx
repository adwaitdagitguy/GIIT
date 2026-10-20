import React, { useMemo, useCallback } from 'react'
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  Node,
  Edge,
  MarkerType,
  BackgroundVariant
} from '@xyflow/react'
import { useWorkbenchStore } from '../../store/workbenchStore'
import { PhysicsNode } from './PhysicsNode'
import { Network, Play, Info } from 'lucide-react'

const nodeTypes = {
  physicsNode: PhysicsNode
}

export const GraphCanvas: React.FC = () => {
  const { graph, selectedNodeId, setSelectedNodeId, loadDemoModel } = useWorkbenchStore()

  // Layout algorithm: assigns positions in hierarchical columns based on node type
  const { initialNodes, initialEdges } = useMemo(() => {
    if (!graph || !graph.nodes || graph.nodes.length === 0) {
      return { initialNodes: [], initialEdges: [] }
    }

    const typeColumns: Record<string, number> = {
      input: 0,
      latent: 1,
      parameter: 1,
      intermediate: 2,
      output: 3,
      physics_loss: 4,
      state: 3
    }

    const columnCounts: Record<number, number> = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0 }

    const nodes: Node[] = graph.nodes.map((n) => {
      const col = typeColumns[n.type] ?? 2
      const row = columnCounts[col] || 0
      columnCounts[col] = (columnCounts[col] || 0) + 1

      const x = 50 + col * 240
      const y = 50 + row * 130

      return {
        id: n.id,
        type: 'physicsNode',
        position: { x, y },
        selected: n.id === selectedNodeId,
        data: {
          label: n.label,
          symbol: n.symbol,
          type: n.type,
          formula: n.formula,
          unit: n.unit,
          description: n.description,
          defaultValue: n.default_value
        }
      }
    })

    const edges: Edge[] = (graph.edges || []).map((e, idx) => ({
      id: `e-${e.source}-${e.target}-${idx}`,
      source: e.source,
      target: e.target,
      label: e.label,
      animated: e.source.includes('latent') || e.target.includes('residual'),
      style: {
        stroke: e.target.includes('residual') ? '#f43f5e' : '#38bdf8',
        strokeWidth: 2
      },
      labelStyle: {
        fill: '#94a3b8',
        fontWeight: 600,
        fontSize: 10
      },
      labelBgStyle: {
        fill: '#0f172a',
        fillOpacity: 0.85
      },
      labelBgPadding: [4, 2],
      labelBgBorderRadius: 4,
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: e.target.includes('residual') ? '#f43f5e' : '#38bdf8',
        width: 16,
        height: 16
      }
    }))

    return { initialNodes: nodes, initialEdges: edges }
  }, [graph, selectedNodeId])

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges)

  // Keep nodes in sync when graph or selectedNodeId changes
  React.useEffect(() => {
    setNodes(initialNodes)
    setEdges(initialEdges)
  }, [initialNodes, initialEdges, setNodes, setEdges])

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      setSelectedNodeId(node.id)
    },
    [setSelectedNodeId]
  )

  const onPaneClick = useCallback(() => {
    setSelectedNodeId(null)
  }, [setSelectedNodeId])

  if (!graph || graph.nodes.length === 0) {
    return (
      <div className="flex-1 h-full flex flex-col items-center justify-center p-8 text-center bg-slate-950/80 select-none">
        <div className="w-16 h-16 rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-center mb-4 shadow-xl text-cyan-400">
          <Network className="w-8 h-8 opacity-60" />
        </div>
        <h3 className="text-base font-bold text-slate-200 mb-1">
          No Physics Graph Loaded
        </h3>
        <p className="text-xs text-slate-400 max-w-md mb-6 leading-relaxed">
          Upload your PyTorch model (.py) and weights (.pt) on the left sidebar, or load the pre-trained Damped Oscillator demo to visualize the causal physics graph.
        </p>
        <button
          onClick={loadDemoModel}
          className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white text-xs font-semibold rounded-lg shadow-lg shadow-cyan-500/20 transition-all cursor-pointer"
        >
          <Play className="w-4 h-4 fill-current" />
          <span>Load Damped Oscillator Demo</span>
        </button>
      </div>
    )
  }

  return (
    <div className="flex-1 h-full relative bg-slate-950">
      {/* Top Legend Bar */}
      <div className="absolute top-3 left-3 z-10 flex flex-wrap items-center gap-1.5 p-1.5 bg-slate-900/90 border border-slate-800/90 rounded-lg backdrop-blur-md text-[10px] select-none shadow-lg">
        <span className="text-slate-400 font-semibold px-1 flex items-center gap-1">
          <Info className="w-3 h-3 text-cyan-400" /> Legend:
        </span>
        <span className="px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-300 border border-blue-500/30">
          Input (t)
        </span>
        <span className="px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-300 border border-purple-500/30">
          Latent Layer
        </span>
        <span className="px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
          ODE Param (μ, k)
        </span>
        <span className="px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">
          Derivative (ẏ, ÿ)
        </span>
        <span className="px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-300 border border-rose-500/30">
          Loss Residual (f)
        </span>
        <span className="px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
          Output (y)
        </span>
      </div>

      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        nodeTypes={nodeTypes}
        fitView
        minZoom={0.2}
        maxZoom={2.5}
        className="bg-slate-950"
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={18}
          size={1.2}
          color="#1e293b"
        />
        <Controls position="bottom-right" className="!mb-3 !mr-3" />
        <MiniMap
          nodeColor={(n: any) => {
            if (n.data?.type === 'input') return '#3b82f6'
            if (n.data?.type === 'latent') return '#a855f7'
            if (n.data?.type === 'parameter') return '#06b6d4'
            if (n.data?.type === 'intermediate') return '#f59e0b'
            if (n.data?.type === 'physics_loss') return '#f43f5e'
            if (n.data?.type === 'output') return '#10b981'
            return '#64748b'
          }}
          maskColor="rgba(3, 7, 18, 0.75)"
          position="bottom-left"
          className="!mb-3 !ml-3 !w-32 !h-24"
        />
      </ReactFlow>
    </div>
  )
}
