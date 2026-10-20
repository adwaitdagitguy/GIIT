import React from 'react'
import Plot from 'react-plotly.js'
import { TrajectoryData } from '../../types/api'
import { Activity, Play } from 'lucide-react'

interface TrajectoryPlotProps {
  data: TrajectoryData | null
  layerName?: string
  interventionType?: string
  strength?: number
}

export const TrajectoryPlot: React.FC<TrajectoryPlotProps> = ({
  data,
  layerName = "fch.0.weight",
  interventionType = "noise",
  strength = 0.35
}) => {
  if (!data || !data.xValues || data.xValues.length === 0) {
    return (
      <div className="h-full w-full flex flex-col items-center justify-center text-slate-500 p-6 text-center select-none">
        <Activity className="w-8 h-8 opacity-40 text-cyan-400 mb-2" />
        <p className="text-xs text-slate-300 font-semibold">No Trajectory Computed</p>
        <p className="text-[11px] text-slate-500">
          Run an intervention in the right panel to simulate ODE output divergence.
        </p>
      </div>
    )
  }

  const traces: any[] = [
    {
      x: data.xValues,
      y: data.yBaseline,
      type: 'scatter',
      mode: 'lines',
      name: 'Baseline PINN',
      line: {
        color: '#38bdf8',
        width: 2.5
      }
    },
    {
      x: data.xValues,
      y: data.yIntervened,
      type: 'scatter',
      mode: 'lines',
      name: `Intervened (${interventionType} @ α=${strength})`,
      line: {
        color: '#f43f5e',
        width: 2.5,
        dash: 'dash'
      }
    }
  ]

  if (data.yGroundTruth && data.yGroundTruth.length > 0) {
    traces.unshift({
      x: data.xValues,
      y: data.yGroundTruth,
      type: 'scatter',
      mode: 'lines',
      name: 'Analytical Ground Truth',
      line: {
        color: '#10b981',
        width: 1.5,
        dash: 'dot'
      }
    })
  }

  return (
    <div className="w-full h-full relative p-2">
      <Plot
        data={traces}
        layout={{
          autosize: true,
          margin: { l: 45, r: 20, t: 25, b: 35 },
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
            title: { text: 'Time (t) [s]', font: { color: '#94a3b8', size: 10 } },
            color: '#94a3b8',
            gridcolor: 'rgba(51, 65, 85, 0.4)',
            zerolinecolor: 'rgba(51, 65, 85, 0.7)',
            tickfont: { size: 9 }
          },
          yaxis: {
            title: { text: 'Displacement y(t) [m]', font: { color: '#94a3b8', size: 10 } },
            color: '#94a3b8',
            gridcolor: 'rgba(51, 65, 85, 0.4)',
            zerolinecolor: 'rgba(51, 65, 85, 0.7)',
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
