import React from 'react'
import { ErrorMetrics } from '../../types/api'
import { formatScientific, formatPercent } from '../../lib/utils'
import {
  TrendingUp,
  AlertTriangle,
  Activity,
  CheckCircle,
  Zap,
  BarChart2
} from 'lucide-react'

interface MetricsPanelProps {
  metrics: ErrorMetrics | null
}

export const MetricsPanel: React.FC<MetricsPanelProps> = ({ metrics }) => {
  if (!metrics) {
    return (
      <div className="h-full flex items-center justify-center p-6 text-slate-500 text-xs text-center select-none">
        Run an intervention to calculate baseline vs perturbed error metrics.
      </div>
    )
  }

  const metricCards = [
    {
      label: 'Intervened MSE',
      value: formatScientific(metrics.mseIntervened),
      subtext: `Baseline: ${formatScientific(metrics.mseBaseline)}`,
      badge: metrics.mseIntervened > 0.01 ? 'Perturbed' : 'Stable',
      badgeColor: metrics.mseIntervened > 0.01 ? 'text-amber-400 bg-amber-950/50' : 'text-emerald-400 bg-emerald-950/50'
    },
    {
      label: 'Trajectory RMSE',
      value: formatScientific(metrics.rmse),
      subtext: `MAE: ${formatScientific(metrics.mae)}`,
      badge: 'Root-Mean-Sq',
      badgeColor: 'text-blue-400 bg-blue-950/50'
    },
    {
      label: 'L2 Norm Perturbed',
      value: metrics.l2Intervened.toFixed(4),
      subtext: `Baseline L2: ${metrics.l2Baseline.toFixed(4)}`,
      badge: `Δ: ${(metrics.l2Intervened - metrics.l2Baseline).toFixed(4)}`,
      badgeColor: 'text-purple-400 bg-purple-950/50'
    },
    {
      label: 'Max Trajectory Divergence',
      value: metrics.maxDivergence.toFixed(4),
      subtext: 'Peak absolute deviation',
      badge: 'Max Dev',
      badgeColor: 'text-rose-400 bg-rose-950/50'
    },
    {
      label: 'Performance Degradation',
      value: `+${metrics.percentDegradation.toFixed(1)}%`,
      subtext: 'Loss increase relative to baseline',
      badge: 'Degradation',
      badgeColor: 'text-orange-400 bg-orange-950/50'
    }
  ]

  return (
    <div className="p-3 h-full overflow-y-auto select-none">
      <div className="grid grid-cols-2 md:grid-cols-5 gap-2.5 h-full items-center">
        {metricCards.map((c, i) => (
          <div
            key={i}
            className="bg-slate-950/70 border border-slate-800/90 rounded-xl p-3 flex flex-col justify-between shadow-md hover:border-slate-700 transition-all"
          >
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
                {c.label}
              </span>
              <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded border border-current/20 ${c.badgeColor}`}>
                {c.badge}
              </span>
            </div>
            <div className="text-base font-bold font-mono text-slate-100 my-0.5">
              {c.value}
            </div>
            <div className="text-[10px] text-slate-400 truncate">
              {c.subtext}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
