import React, { useEffect } from 'react'
import { useWorkbenchStore } from '../../store/workbenchStore'
import { api } from '../../services/api'
import {
  Activity,
  Sparkles,
  RotateCcw,
  Wifi,
  WifiOff,
  PlayCircle,
  HelpCircle,
  Cpu
} from 'lucide-react'

export const TopBar: React.FC = () => {
  const {
    sessionId,
    modelMetadata,
    backendConnected,
    setBackendConnected,
    isMockMode,
    setIsMockMode,
    loadDemoModel,
    resetWorkspace,
    setIsTourActive
  } = useWorkbenchStore()

  // Heartbeat check for backend
  useEffect(() => {
    let mounted = true
    const checkServer = async () => {
      const ok = await api.checkHealth()
      if (mounted) setBackendConnected(ok)
    }
    checkServer()
    const interval = setInterval(checkServer, 10000)
    return () => {
      mounted = false
      clearInterval(interval)
    }
  }, [setBackendConnected])

  return (
    <header className="h-14 bg-slate-900/90 border-b border-slate-800 px-4 flex items-center justify-between z-30 select-none backdrop-blur-md">
      {/* Left: Branding & App Title */}
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-blue-600 via-cyan-500 to-indigo-500 flex items-center justify-center shadow-lg shadow-blue-500/20">
          <Activity className="w-5 h-5 text-white" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-sm font-bold text-slate-100 tracking-tight">
              SciML Intervention Workbench
            </h1>
            <span className="text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20 tracking-wider">
              PINN Studio
            </span>
          </div>
          <p className="text-[11px] text-slate-400 hidden sm:block">
            Physics-Informed Causal Graph & Neural Layer Perturbation Analyzer
          </p>
        </div>
      </div>

      {/* Center: Active Session & Model Info */}
      <div className="hidden md:flex items-center gap-2 px-3 py-1 bg-slate-950/60 rounded-full border border-slate-800/80">
        <Cpu className="w-3.5 h-3.5 text-cyan-400" />
        <span className="text-xs text-slate-300 font-medium">
          {modelMetadata ? (
            <>
              <span className="text-cyan-400 font-semibold">{modelMetadata.modelClass}</span>
              <span className="text-slate-500 mx-1.5">•</span>
              <span className="text-slate-400">{modelMetadata.totalParameters} params</span>
            </>
          ) : (
            <span className="text-slate-500 italic">No PINN Model Loaded</span>
          )}
        </span>
        {sessionId && (
          <span className="text-[10px] font-mono text-slate-500 bg-slate-900 px-1.5 py-0.5 rounded border border-slate-800">
            {sessionId.slice(0, 16)}
          </span>
        )}
      </div>

      {/* Right: Actions & Server Health */}
      <div className="flex items-center gap-2">
        {/* Quick Demo Button if no model */}
        {!modelMetadata && (
          <button
            onClick={loadDemoModel}
            className="flex items-center gap-1.5 px-2.5 py-1 text-xs font-semibold text-cyan-300 bg-cyan-950/60 hover:bg-cyan-900/60 border border-cyan-500/30 rounded-md transition-all shadow-sm shadow-cyan-500/10 hover:shadow-cyan-500/25 cursor-pointer"
          >
            <PlayCircle className="w-3.5 h-3.5 text-cyan-400" />
            <span>Load Demo (Oscillator)</span>
          </button>
        )}

        {/* Backend Health Badge */}
        <div
          className={`flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px] font-medium border ${
            backendConnected
              ? 'bg-emerald-950/40 text-emerald-400 border-emerald-500/30'
              : 'bg-amber-950/30 text-amber-400 border-amber-500/30'
          }`}
          title={backendConnected ? "Connected to FastAPI backend (localhost:8000)" : "Backend offline — running in realistic mock physics mode"}
        >
          {backendConnected ? (
            <>
              <Wifi className="w-3 h-3 text-emerald-400" />
              <span className="hidden sm:inline">Backend Live</span>
            </>
          ) : (
            <>
              <WifiOff className="w-3 h-3 text-amber-400" />
              <span className="hidden sm:inline">Mock Mode</span>
            </>
          )}
        </div>

        {/* Start Tour */}
        <button
          onClick={() => setIsTourActive(true)}
          className="flex items-center gap-1 px-2.5 py-1 text-xs font-medium text-slate-300 hover:text-white bg-slate-800/80 hover:bg-slate-700/80 border border-slate-700 rounded-md transition-colors cursor-pointer"
          title="Start Interactive Guided Tour"
        >
          <Sparkles className="w-3.5 h-3.5 text-amber-400" />
          <span className="hidden sm:inline">Tour</span>
        </button>

        {/* Reset */}
        {modelMetadata && (
          <button
            onClick={resetWorkspace}
            className="p-1.5 text-slate-400 hover:text-rose-400 hover:bg-slate-800 rounded-md transition-colors cursor-pointer"
            title="Reset Workspace"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        )}
      </div>
    </header>
  )
}
