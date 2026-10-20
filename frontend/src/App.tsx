import React from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { TopBar } from './components/layout/TopBar'
import { Sidebar } from './components/layout/Sidebar'
import { GraphCanvas } from './components/graph/GraphCanvas'
import { NodeInspector } from './components/inspector/NodeInspector'
import { InterventionControls } from './components/inspector/InterventionControls'
import { AnalyticsDock } from './components/analytics/AnalyticsDock'
import { GuidedTour } from './components/onboarding/GuidedTour'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: false
    }
  }
})

export const App: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="flex flex-col h-screen w-screen bg-slate-950 text-slate-100 overflow-hidden font-sans select-none">
        {/* Top Header Navigation */}
        <TopBar />

        {/* Main 3-Column Studio Workspace */}
        <main className="flex-1 flex min-h-0 overflow-hidden relative">
          {/* Left Panel: Model & Ruleset Uploader */}
          <Sidebar />

          {/* Center Pane: React Flow Causal Physics Graph */}
          <div className="flex-1 h-full min-w-0 relative">
            <GraphCanvas />
          </div>

          {/* Right Panel: Node Inspector & Intervention Studio */}
          <aside className="w-96 h-full bg-slate-900/60 border-l border-slate-800 flex flex-col divide-y divide-slate-800 overflow-y-auto z-10">
            <NodeInspector />
            <InterventionControls />
          </aside>
        </main>

        {/* Bottom Panel: Multi-Tab Analytics Dock */}
        <AnalyticsDock />

        {/* Interactive Guided Tour Modal */}
        <GuidedTour />
      </div>
    </QueryClientProvider>
  )
}

export default App
