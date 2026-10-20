import { create } from 'zustand'
import {
  ParsedGraph,
  ModelMetadata,
  InterventionResult,
  SweepResult,
  FullAnalysisResult,
  NodeLayerMapping,
  InterventionType,
  AssistantMessage
} from '../types/api'
import {
  mockParsedGraph,
  mockModelMetadata,
  createMockInterventionResult,
  createMockSweepResult,
  mockFullAnalysis,
  mockMappingList
} from '../mocks/fixtures'

interface WorkbenchState {
  sessionId: string | null
  backendConnected: boolean
  isMockMode: boolean
  modelMetadata: ModelMetadata | null
  graph: ParsedGraph | null
  selectedNodeId: string | null

  // Intervention Config & Output
  selectedLayer: string
  interventionType: InterventionType
  interventionStrength: number
  interventionResult: InterventionResult | null
  sweepResult: SweepResult | null

  // Analysis & Mapping
  fullAnalysis: FullAnalysisResult | null
  mapping: NodeLayerMapping[] | null

  // UI State
  activeAnalyticsTab: 'trajectory' | 'metrics' | 'sensitivity' | 'mapping' | 'sweep' | 'assistant'
  isInterventionRunning: boolean
  isSweepRunning: boolean
  isAnalyzing: boolean
  isMappingRunning: boolean
  isTourActive: boolean
  chatMessages: AssistantMessage[]

  // Actions
  setBackendConnected: (status: boolean) => void
  setIsMockMode: (mock: boolean) => void
  setModelData: (metadata: ModelMetadata, graph?: ParsedGraph) => void
  setGraph: (graph: ParsedGraph) => void
  setSelectedNodeId: (nodeId: string | null) => void
  setSelectedLayer: (layer: string) => void
  setInterventionType: (type: InterventionType) => void
  setInterventionStrength: (strength: number) => void
  setInterventionResult: (res: InterventionResult | null) => void
  setSweepResult: (res: SweepResult | null) => void
  setFullAnalysis: (res: FullAnalysisResult | null) => void
  setMapping: (mapping: NodeLayerMapping[] | null) => void
  setActiveAnalyticsTab: (tab: WorkbenchState['activeAnalyticsTab']) => void
  setIsInterventionRunning: (loading: boolean) => void
  setIsSweepRunning: (loading: boolean) => void
  setIsAnalyzing: (loading: boolean) => void
  setIsMappingRunning: (loading: boolean) => void
  setIsTourActive: (active: boolean) => void
  addChatMessage: (msg: AssistantMessage) => void
  loadDemoModel: () => void
  resetWorkspace: () => void
}

export const useWorkbenchStore = create<WorkbenchState>((set) => ({
  sessionId: null,
  backendConnected: false,
  isMockMode: false,
  modelMetadata: null,
  graph: null,
  selectedNodeId: null,

  selectedLayer: "fch.0.weight",
  interventionType: "noise",
  interventionStrength: 0.35,
  interventionResult: null,
  sweepResult: null,

  fullAnalysis: null,
  mapping: null,

  activeAnalyticsTab: 'trajectory',
  isInterventionRunning: false,
  isSweepRunning: false,
  isAnalyzing: false,
  isMappingRunning: false,
  isTourActive: false,
  chatMessages: [
    {
      id: "1",
      role: "assistant",
      content: "Welcome to the SciML Intervention Workbench. Load a PINN model or click 'Load Demo Model' to inspect the physics graph and test layer interventions.",
      timestamp: new Date().toLocaleTimeString()
    }
  ],

  setBackendConnected: (status) => set({ backendConnected: status }),
  setIsMockMode: (mock) => set({ isMockMode: mock }),

  setModelData: (metadata, graph) =>
    set((state) => ({
      sessionId: metadata.sessionId,
      modelMetadata: metadata,
      graph: graph || state.graph || mockParsedGraph,
      selectedLayer: metadata.perturbableTensors[0] || "fch.0.weight",
      // Initialize an initial baseline trajectory
      interventionResult: createMockInterventionResult(
        metadata.perturbableTensors[0] || "fch.0.weight",
        "noise",
        0.0
      )
    })),

  setGraph: (graph) => set({ graph }),
  setSelectedNodeId: (nodeId) => set({ selectedNodeId: nodeId }),
  setSelectedLayer: (layer) => set({ selectedLayer: layer }),
  setInterventionType: (type) => set({ interventionType: type }),
  setInterventionStrength: (strength) => set({ interventionStrength: strength }),
  setInterventionResult: (res) => set({ interventionResult: res }),
  setSweepResult: (res) => set({ sweepResult: res }),
  setFullAnalysis: (res) => set({ fullAnalysis: res }),
  setMapping: (mapping) => set({ mapping }),
  setActiveAnalyticsTab: (tab) => set({ activeAnalyticsTab: tab }),
  setIsInterventionRunning: (loading) => set({ isInterventionRunning: loading }),
  setIsSweepRunning: (loading) => set({ isSweepRunning: loading }),
  setIsAnalyzing: (loading) => set({ isAnalyzing: loading }),
  setIsMappingRunning: (loading) => set({ isMappingRunning: loading }),
  setIsTourActive: (active) => set({ isTourActive: active }),

  addChatMessage: (msg) =>
    set((state) => ({ chatMessages: [...state.chatMessages, msg] })),

  loadDemoModel: () => {
    const demoRes = createMockInterventionResult("fch.0.weight", "noise", 0.35)
    set({
      sessionId: "demo_damped_oscillator",
      modelMetadata: mockModelMetadata,
      graph: mockParsedGraph,
      selectedNodeId: "node_mu",
      selectedLayer: "fch.0.weight",
      interventionType: "noise",
      interventionStrength: 0.35,
      interventionResult: demoRes,
      sweepResult: createMockSweepResult("fch.0.weight", "noise"),
      fullAnalysis: mockFullAnalysis,
      mapping: mockMappingList,
      activeAnalyticsTab: "trajectory"
    })
  },

  resetWorkspace: () =>
    set({
      sessionId: null,
      modelMetadata: null,
      graph: null,
      selectedNodeId: null,
      interventionResult: null,
      sweepResult: null,
      fullAnalysis: null,
      mapping: null,
      activeAnalyticsTab: "trajectory"
    })
}))
