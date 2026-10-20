import React, { useState, useRef } from 'react'
import { useWorkbenchStore } from '../../store/workbenchStore'
import { api } from '../../services/api'
import {
  Upload,
  FileCode,
  HardDrive,
  Network,
  Sparkles,
  CheckCircle2,
  AlertCircle,
  Play,
  Layers,
  FileJson,
  X
} from 'lucide-react'

export const Sidebar: React.FC = () => {
  const {
    modelMetadata,
    graph,
    sessionId,
    setModelData,
    setGraph,
    loadDemoModel
  } = useWorkbenchStore()

  // File states
  const [modelFile, setModelFile] = useState<File | null>(null)
  const [weightsFile, setWeightsFile] = useState<File | null>(null)
  const [rulesetFile, setRulesetFile] = useState<File | null>(null)

  const [isLoading, setIsLoading] = useState(false)
  const [isGeneratingRuleset, setIsGeneratingRuleset] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [llmPrompt, setLlmPrompt] = useState("")
  const [showLlmModal, setShowLlmModal] = useState(false)

  // File input refs
  const modelInputRef = useRef<HTMLInputElement>(null)
  const weightsInputRef = useRef<HTMLInputElement>(null)
  const rulesetInputRef = useRef<HTMLInputElement>(null)

  // Handle upload submit
  const handleLoadModel = async () => {
    if (!modelFile || !weightsFile) {
      setErrorMessage("Please select both Model (.py) and Weights (.pt) files.")
      return
    }

    setIsLoading(true)
    setErrorMessage(null)

    try {
      // Upload model, weights, and optional ruleset in one shot
      const { metadata, graph: loadedGraph } = await api.uploadModel(modelFile, weightsFile, rulesetFile)
      setModelData(metadata, loadedGraph)
    } catch (err: any) {
      console.error("Load model error:", err)
      setErrorMessage(err.message || "Failed to load model. Please verify file formats.")
    } finally {
      setIsLoading(false)
    }
  }

  // Handle LLM Ruleset Generation
  const handleGenerateRuleset = async () => {
    if (!sessionId) {
      setErrorMessage("Please load a model first before generating ruleset.")
      return
    }

    setIsGeneratingRuleset(true)
    setShowLlmModal(false)

    try {
      const generated = await api.generateRuleset(sessionId, llmPrompt)
      setGraph(generated)
    } catch (err: any) {
      setErrorMessage("LLM Generation failed: " + err.message)
    } finally {
      setIsGeneratingRuleset(false)
    }
  }

  return (
    <aside className="w-80 h-full bg-slate-900/60 border-r border-slate-800/80 flex flex-col justify-between p-3 select-none overflow-y-auto">
      <div className="space-y-4">
        {/* Section Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-cyan-400" />
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-200">
              PINN Model & Ruleset
            </h2>
          </div>
          <span className="text-[10px] text-slate-400 bg-slate-800/60 px-1.5 py-0.5 rounded border border-slate-700/60">
            Step 1
          </span>
        </div>

        {/* Error notification */}
        {errorMessage && (
          <div className="p-2.5 rounded-lg bg-rose-950/40 border border-rose-500/30 text-rose-300 text-xs flex items-start gap-2">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="font-medium">{errorMessage}</p>
            </div>
            <button onClick={() => setErrorMessage(null)} className="text-rose-400 hover:text-rose-200">
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        )}

        {/* Upload Zone 1: Model Definition (.py) */}
        <div className="space-y-1">
          <label className="text-[11px] font-semibold text-slate-300 flex items-center justify-between">
            <span className="flex items-center gap-1.5">
              <FileCode className="w-3.5 h-3.5 text-blue-400" />
              1. Model Architecture (.py)
            </span>
            {modelFile && <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />}
          </label>
          <input
            type="file"
            ref={modelInputRef}
            accept=".py"
            className="hidden"
            onChange={(e) => setModelFile(e.target.files?.[0] || null)}
          />
          <div
            onClick={() => modelInputRef.current?.click()}
            className={`p-2.5 rounded-lg border border-dashed transition-all cursor-pointer text-center ${
              modelFile
                ? 'bg-blue-950/30 border-blue-500/50 text-blue-200'
                : 'bg-slate-950/50 hover:bg-slate-800/40 border-slate-700/70 text-slate-400 hover:border-slate-500'
            }`}
          >
            {modelFile ? (
              <div className="flex items-center justify-between text-xs">
                <span className="truncate font-mono font-medium">{modelFile.name}</span>
                <span className="text-[10px] text-slate-400">{(modelFile.size / 1024).toFixed(1)} KB</span>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-1">
                <Upload className="w-4 h-4 text-slate-500 mb-1" />
                <span className="text-xs font-medium text-slate-300">Choose .py model class</span>
                <span className="text-[10px] text-slate-400">PyTorch nn.Module architecture</span>
              </div>
            )}
          </div>
        </div>

        {/* Upload Zone 2: Model Weights (.pt) */}
        <div className="space-y-1">
          <label className="text-[11px] font-semibold text-slate-300 flex items-center justify-between">
            <span className="flex items-center gap-1.5">
              <HardDrive className="w-3.5 h-3.5 text-purple-400" />
              2. Trained Weights (.pt / .pth)
            </span>
            {weightsFile && <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />}
          </label>
          <input
            type="file"
            ref={weightsInputRef}
            accept=".pt,.pth"
            className="hidden"
            onChange={(e) => setWeightsFile(e.target.files?.[0] || null)}
          />
          <div
            onClick={() => weightsInputRef.current?.click()}
            className={`p-2.5 rounded-lg border border-dashed transition-all cursor-pointer text-center ${
              weightsFile
                ? 'bg-purple-950/30 border-purple-500/50 text-purple-200'
                : 'bg-slate-950/50 hover:bg-slate-800/40 border-slate-700/70 text-slate-400 hover:border-slate-500'
            }`}
          >
            {weightsFile ? (
              <div className="flex items-center justify-between text-xs">
                <span className="truncate font-mono font-medium">{weightsFile.name}</span>
                <span className="text-[10px] text-slate-400">{(weightsFile.size / 1024).toFixed(1)} KB</span>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-1">
                <Upload className="w-4 h-4 text-slate-500 mb-1" />
                <span className="text-xs font-medium text-slate-300">Choose .pt checkpoint</span>
                <span className="text-[10px] text-slate-400">State dictionary or full model</span>
              </div>
            )}
          </div>
        </div>

        {/* Upload Zone 3: Physics Graph Ruleset (.json) */}
        <div className="space-y-1">
          <label className="text-[11px] font-semibold text-slate-300 flex items-center justify-between">
            <span className="flex items-center gap-1.5">
              <Network className="w-3.5 h-3.5 text-emerald-400" />
              3. Physics Graph (.json) <span className="text-slate-400 font-normal">(Optional)</span>
            </span>
            {rulesetFile && <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />}
          </label>
          <input
            type="file"
            ref={rulesetInputRef}
            accept=".json"
            className="hidden"
            onChange={(e) => setRulesetFile(e.target.files?.[0] || null)}
          />
          <div
            onClick={() => rulesetInputRef.current?.click()}
            className={`p-2.5 rounded-lg border border-dashed transition-all cursor-pointer text-center ${
              rulesetFile
                ? 'bg-emerald-950/30 border-emerald-500/50 text-emerald-200'
                : 'bg-slate-950/50 hover:bg-slate-800/40 border-slate-700/70 text-slate-400 hover:border-slate-500'
            }`}
          >
            {rulesetFile ? (
              <div className="flex items-center justify-between text-xs">
                <span className="truncate font-mono font-medium">{rulesetFile.name}</span>
                <span className="text-[10px] text-slate-400">{(rulesetFile.size / 1024).toFixed(1)} KB</span>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-1">
                <FileJson className="w-4 h-4 text-slate-500 mb-1" />
                <span className="text-xs font-medium text-slate-300">Upload graph.json</span>
                <span className="text-[10px] text-slate-400">ODE/PDE causal ruleset</span>
              </div>
            )}
          </div>
        </div>

        {/* Action Buttons */}
        <div className="pt-2 space-y-2">
          <button
            onClick={handleLoadModel}
            disabled={isLoading || (!modelFile && !weightsFile)}
            className="w-full py-2.5 px-3 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 disabled:opacity-40 text-white text-xs font-semibold rounded-lg shadow-md shadow-blue-500/20 transition-all flex items-center justify-center gap-2 cursor-pointer disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <>
                <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>Instantiating PINN...</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>Initialize Workbench</span>
              </>
            )}
          </button>

          {/* AI Auto-generate button */}
          <button
            onClick={() => setShowLlmModal(true)}
            disabled={isGeneratingRuleset}
            className="w-full py-2 px-3 bg-slate-800/80 hover:bg-slate-700/80 border border-slate-700 text-slate-200 text-xs font-medium rounded-lg transition-all flex items-center justify-center gap-2 cursor-pointer"
          >
            <Sparkles className="w-3.5 h-3.5 text-amber-400" />
            <span>Auto-Generate Ruleset (AI)</span>
          </button>
        </div>

        {/* Loaded Model Summary Card */}
        {modelMetadata && graph && (
          <div className="p-3 bg-slate-950/70 border border-slate-800 rounded-lg space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold text-slate-300">Physics System</span>
              <span className="text-[10px] font-mono text-cyan-400 bg-cyan-950/60 px-1.5 py-0.5 rounded border border-cyan-500/30">
                {graph.metadata.system}
              </span>
            </div>
            <div className="text-xs font-mono text-slate-300 bg-slate-900/90 p-1.5 rounded border border-slate-800">
              {graph.metadata.equation}
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              {graph.metadata.description}
            </p>
            <div className="grid grid-cols-2 gap-1.5 pt-1 text-[10px]">
              <div className="bg-slate-900/60 p-1.5 rounded border border-slate-800">
                <span className="text-slate-400 block">Nodes:</span>
                <span className="font-semibold text-slate-200">{graph.nodes.length} physical</span>
              </div>
              <div className="bg-slate-900/60 p-1.5 rounded border border-slate-800">
                <span className="text-slate-400 block">Layers:</span>
                <span className="font-semibold text-slate-200">{modelMetadata.perturbableTensors.length} perturbable</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer / Quick Demo */}
      <div className="pt-3 border-t border-slate-800/80 text-center space-y-2">
        <p className="text-[10px] text-slate-400">Need instant testing without uploading files?</p>
        <button
          onClick={loadDemoModel}
          className="w-full py-1.5 text-xs text-cyan-400 hover:text-cyan-300 bg-cyan-950/30 hover:bg-cyan-900/40 border border-cyan-500/20 rounded-md transition-colors font-medium cursor-pointer"
        >
          Load Damped Oscillator Demo
        </button>
      </div>

      {/* LLM Generation Modal */}
      {showLlmModal && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-700 rounded-xl max-w-md w-full p-5 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-amber-400" />
                <h3 className="text-sm font-bold text-slate-100">Auto-Generate Physics Ruleset</h3>
              </div>
              <button onClick={() => setShowLlmModal(false)} className="text-slate-400 hover:text-white">
                <X className="w-4 h-4" />
              </button>
            </div>

            <p className="text-xs text-slate-300 leading-relaxed">
              Uses OpenRouter LLM to parse your PINN architecture code, extract governing ODE/PDE equations, identify parameters (damping, stiffness), and construct a verified causal graph.
            </p>

            <textarea
              value={llmPrompt}
              onChange={(e) => setLlmPrompt(e.target.value)}
              placeholder="Custom instructions (e.g. 'Target 2nd-order damped harmonic oscillator with mu and k parameters')..."
              className="w-full h-24 bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:border-cyan-500"
            />

            <div className="flex justify-end gap-2 pt-2">
              <button
                onClick={() => setShowLlmModal(false)}
                className="px-3 py-1.5 text-xs text-slate-400 hover:text-white"
              >
                Cancel
              </button>
              <button
                onClick={handleGenerateRuleset}
                disabled={isGeneratingRuleset}
                className="px-4 py-1.5 bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-500 hover:to-orange-500 text-white text-xs font-semibold rounded-lg shadow transition-all cursor-pointer"
              >
                {isGeneratingRuleset ? "Generating via LLM..." : "Generate Graph JSON"}
              </button>
            </div>
          </div>
        </div>
      )}
    </aside>
  )
}
