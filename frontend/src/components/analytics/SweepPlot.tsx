import React from 'react'
import Plot from 'react-plotly.js'
import { SweepResult } from '../../types/api'
import { LineChart, Zap } from 'lucide-react'

interface SweepPlotProps {
  data: SweepResult | null
}

export const SweepPlot: React.FC<SweepPlotProps> = ({ data }) => {
  if (!data || !data.strengths || data.strengths.length === 0) {
    return (
      <div className="h-full w-full flex flex-col items-center justify-center text-slate-500 p-6 text-center select-none">
        <LineChart className="w-8 h-8 opacity-40 text-amber-400 mb-2" />
        <p className="text-xs text-slate-300 font-semibold">No Parameter Sweep Data</p>
        <p className="text-[11px] text-slate-500">
          Click "Run Parameter Sweep Loop" in the right panel to visualize monotonic degradation across strength levels (0.0 → 1.0).
        </p>
      </div>
    )
  }

  const traces: any[] = [
    {
      x: data.strengths,
      y: data.mseValues,
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Intervention MSE Loss',
      line: { color: '#f59e0b', width: 2.5 },
      marker: { color: '#f59e0b', size: 6 }
    },
    {
      x: data.strengths,
      y: data.maxDivergences,
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Max Trajectory Divergence',
      yaxis: 'y2',
      line: { color: '#f43f5e', width: 2, dash: 'dash' },
      marker: { color: '#f43f5e', size: 5 }
    }
  ]

  return (
    <div className="w-full h-full relative p-2">
      <Plot
        data={traces}
        layout={{
          autosize: true,
          margin: { l: 45, r: 45, t: 25, b: 35 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          showlegend: true,
          legend: {
            orientation: 'h',
            y: 1.15,
            x: 0.5,
            xanchor: 'center',
            font: { color: '#94a3b8', size: 10 }
          },
          xaxis: {
            title: { text: 'Perturbation Strength (α)', font: { color: '#94a3b8', size: 10 } },
            color: '#94a3b8',
            gridcolor: 'rgba(51, 65, 85, 0.4)',
            zerolinecolor: 'rgba(51, 65, 85, 0.7)',
            tickfont: { size: 9 }
          },
          yaxis: {
            title: { text: 'MSE Loss', font: { color: '#f59e0b', size: 10 } },
            color: '#f59e0b',
            gridcolor: 'rgba(51, 65, 85, 0.4)',
            zerolinecolor: 'rgba(51, 65, 85, 0.7)',
            tickfont: { size: 9 }
          },
          yaxis2: {
            title: { text: 'Max Divergence', font: { color: '#f43f5e', size: 10 } },
            color: '#f43f5e',
            overlaying: 'y',
            side: 'right',
            tickfont: { size: 9 }
          }
        }}
        useResizeHandler
        style={{ width: '100%', height: '100%' }}
        config={{
          displayModeBar: false,
          responsive: true
        }}
      />
    </div>
  )
}
