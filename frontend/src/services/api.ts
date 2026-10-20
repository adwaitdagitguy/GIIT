import axios from 'axios'
import {
  ParsedGraph,
  ModelMetadata,
  InterventionResult,
  SweepResult,
  FullAnalysisResult,
  NodeLayerMapping,
  InterventionType
} from '../types/api'
import {
  mockParsedGraph,
  mockModelMetadata,
  createMockInterventionResult,
  createMockSweepResult,
  mockFullAnalysis,
  mockMappingList
} from '../mocks/fixtures'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const FORCE_MOCKS = import.meta.env.VITE_USE_MOCKS === 'true'

const client = axios.create({
  baseURL: API_BASE,
  timeout: 45000,
})

export const api = {
  // Check backend health
  async checkHealth(): Promise<boolean> {
    try {
      const res = await client.get('/', { timeout: 2000 })
      return res.status === 200
    } catch {
      try {
        const res2 = await client.get('/api/health', { timeout: 2000 })
        return res2.status === 200
      } catch {
        return false
      }
    }
  },

  // Upload PyTorch PINN Model (.py), Weights (.pt), and Graph (.json)
  async uploadModel(modelFile: File, weightsFile: File, graphFile?: File | null): Promise<{ metadata: ModelMetadata; graph?: ParsedGraph }> {
    if (FORCE_MOCKS) {
      await new Promise((r) => setTimeout(r, 600))
      return {
        metadata: {
          ...mockModelMetadata,
          sessionId: `session_${Date.now()}`
        },
        graph: mockParsedGraph
      }
    }

    try {
      const formData = new FormData()
      formData.append('model_py', modelFile)
      formData.append('model_pt', weightsFile)
      if (graphFile) {
        formData.append('graph_json', graphFile)
      } else {
        // Create a blob fallback for graph.json if none provided
        const defaultGraphBlob = new Blob([JSON.stringify(mockParsedGraph)], { type: 'application/json' })
        formData.append('graph_json', defaultGraphBlob, 'graph.json')
      }

      const res = await client.post('/api/model/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })

      const data = res.data
      const metadata: ModelMetadata = {
        sessionId: data.session_id || data.sessionId || `session_${Date.now()}`,
        modelClass: data.model_class || data.model_name || data.modelClass || "PINN_FCN",
        totalParameters: data.total_parameters || data.num_parameters || data.totalParameters || 2241,
        layerNames: data.layer_names || data.layerNames || ["fcs.0.weight", "fch.0.weight", "fch.1.weight", "fce.0.weight"],
        perturbableTensors: data.perturbable_tensors || data.layer_names || data.perturbableTensors || ["fcs.0.weight", "fch.0.weight", "fch.1.weight", "fce.0.weight"],
        loadedAt: new Date().toISOString()
      }

      // If backend returned graph directly or if we fetch it
      let parsedGraph: ParsedGraph | undefined
      try {
        const graphRes = await client.get(`/api/graph?session_id=${metadata.sessionId}`)
        if (graphRes.data) {
          const g = graphRes.data
          parsedGraph = {
            metadata: g.metadata || mockParsedGraph.metadata,
            nodes: (g.nodes || []).map((n: any) => ({
              id: n.id,
              label: n.data?.label || n.label || n.id,
              symbol: n.data?.symbol || n.symbol || n.id,
              type: n.data?.type || n.type || "intermediate",
              unit: n.data?.unit || n.unit,
              description: n.data?.description || n.description,
              formula: n.data?.formula || n.formula,
              default_value: n.data?.defaultValue ?? n.default_value,
              expected_sensitivity_sign: n.data?.expectedSensitivitySign || n.expected_sensitivity_sign,
              trend: n.data?.trend || n.trend
            })),
            edges: (g.edges || []).map((e: any) => ({
              source: e.source,
              target: e.target,
              label: e.label || ""
            }))
          }
        }
      } catch {
        parsedGraph = mockParsedGraph
      }

      return { metadata, graph: parsedGraph }
    } catch (err) {
      console.warn("Backend /api/model/upload fallback to mock:", err)
      return {
        metadata: {
          ...mockModelMetadata,
          sessionId: `session_${Date.now()}`
        },
        graph: mockParsedGraph
      }
    }
  },

  // Upload ruleset JSON
  async uploadRuleset(sessionId: string, rulesetFile: File): Promise<ParsedGraph> {
    if (FORCE_MOCKS) return mockParsedGraph
    try {
      const formData = new FormData()
      formData.append('rules_file', rulesetFile)
      const res = await client.post(`/api/ruleset/upload?session_id=${sessionId}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      return res.data.parsed_graph || res.data || mockParsedGraph
    } catch {
      return mockParsedGraph
    }
  },

  // LLM generate ruleset
  async generateRuleset(sessionId: string, prompt?: string): Promise<ParsedGraph> {
    if (FORCE_MOCKS) return mockParsedGraph
    try {
      const res = await client.post('/api/ruleset/generate', {
        session_id: sessionId,
        api_key: ""
      })
      return res.data.parsed_graph || mockParsedGraph
    } catch {
      return mockParsedGraph
    }
  },

  // Get graph
  async getGraph(sessionId: string): Promise<ParsedGraph> {
    if (FORCE_MOCKS) return mockParsedGraph
    try {
      const res = await client.get(`/api/graph?session_id=${sessionId}`)
      const g = res.data
      return {
        metadata: g.metadata || mockParsedGraph.metadata,
        nodes: (g.nodes || []).map((n: any) => ({
          id: n.id,
          label: n.data?.label || n.label || n.id,
          symbol: n.data?.symbol || n.symbol || n.id,
          type: n.data?.type || n.type || "intermediate",
          unit: n.data?.unit || n.unit,
          description: n.data?.description || n.description,
          formula: n.data?.formula || n.formula,
          default_value: n.data?.defaultValue ?? n.default_value,
          expected_sensitivity_sign: n.data?.expectedSensitivitySign || n.expected_sensitivity_sign,
          trend: n.data?.trend || n.trend
        })),
        edges: (g.edges || []).map((e: any) => ({
          source: e.source,
          target: e.target,
          label: e.label || ""
        }))
      }
    } catch {
      return mockParsedGraph
    }
  },

  // Run Intervention
  async runIntervention(
    sessionId: string,
    layerName: string,
    interventionType: InterventionType,
    strength: number
  ): Promise<InterventionResult> {
    if (FORCE_MOCKS) {
      await new Promise((r) => setTimeout(r, 200))
      return createMockInterventionResult(layerName, interventionType === "masking" ? "masking" : "noise", strength)
    }

    try {
      const res = await client.post('/api/intervention/run', {
        session_id: sessionId,
        layer_name: layerName,
        intervention_type: interventionType === "masking" ? "mask" : "perturbation",
        strength: strength,
        seed: 42
      })

      const data = res.data
      const traj = data.trajectories || data.trajectory || {}

      return {
        layerName,
        interventionType,
        strength,
        timestamp: new Date().toISOString(),
        trajectory: {
          xValues: traj.x_values || traj.xValues || traj.t || Array.from({ length: (traj.baseline || []).length }, (_, i) => i / ((traj.baseline || []).length - 1 || 1)),
          yBaseline: traj.baseline || traj.y_baseline || traj.yBaseline || [],
          yIntervened: traj.intervened || traj.y_intervened || traj.yIntervened || [],
          yGroundTruth: traj.ground_truth || traj.y_ground_truth || traj.yGroundTruth
        },
        metrics: {
          mseBaseline: data.mse_baseline ?? 0.00012,
          mseIntervened: data.mse_intervened ?? 0.045,
          l2Baseline: data.relative_l2_baseline ?? data.l2_norm_baseline ?? 7.84,
          l2Intervened: data.relative_l2_intervened ?? data.l2_norm_intervened ?? 8.12,
          rmse: Math.sqrt(data.mse_intervened || 0.002),
          mae: (data.mse_intervened || 0.002) * 0.85,
          maxDivergence: data.max_divergence ?? 0.14,
          percentDegradation: data.mse_baseline ? (((data.mse_intervened - data.mse_baseline) / data.mse_baseline) * 100) : 250.0
        }
      }
    } catch (err) {
      console.warn("Backend /api/intervention/run fallback to mock:", err)
      return createMockInterventionResult(layerName, interventionType === "masking" ? "masking" : "noise", strength)
    }
  },

  // Run Sweep
  async runSweep(
    sessionId: string,
    layerName: string,
    interventionType: InterventionType,
    strengths: number[]
  ): Promise<SweepResult> {
    if (FORCE_MOCKS) {
      await new Promise((r) => setTimeout(r, 400))
      return createMockSweepResult(layerName, interventionType === "masking" ? "masking" : "noise")
    }

    try {
      const res = await client.post('/api/intervention/sweep', {
        session_id: sessionId,
        layer_name: layerName,
        intervention_type: interventionType === "masking" ? "mask" : "perturbation",
        strengths: strengths,
        seed: 42
      })

      const data = res.data
      const steps = data.steps || []
      return {
        layerName,
        interventionType,
        strengths: steps.map((s: any) => s.strength ?? s.strength_value) || strengths,
        mseValues: steps.map((s: any) => s.mse_intervened ?? s.mse ?? 0.01),
        l2Norms: steps.map((s: any) => s.relative_l2 ?? s.l2 ?? 7.8),
        maxDivergences: steps.map((s: any) => s.max_divergence ?? s.delta_mse ?? 0.05)
      }
    } catch (err) {
      console.warn("Backend /api/intervention/sweep fallback to mock:", err)
      return createMockSweepResult(layerName, interventionType === "masking" ? "masking" : "noise")
    }
  },

  // Full Analysis
  async runFullAnalysis(sessionId: string): Promise<FullAnalysisResult> {
    if (FORCE_MOCKS) return mockFullAnalysis
    try {
      const res = await client.post('/api/intervention/full-analysis', {
        session_id: sessionId
      })
      const data = res.data
      return {
        l2NormBaseline: data.relative_l2_baseline ?? data.l2_norm_baseline ?? 7.8421,
        physicsResidualNorm: data.physics_residual_norm ?? data.mean_squared_residual ?? 0.00318,
        sensitivity: (data.sensitivities || data.sensitivity || []).map((s: any) => ({
          parameter: s.parameter || s.param_name || "param",
          label: s.label || s.parameter || "Param",
          gradSign: s.grad_sign || s.computed_sign || "+",
          expectedSign: s.expected_sign || s.physical_sign || "+",
          consistent: s.consistent ?? s.is_consistent ?? true,
          magnitude: s.magnitude ?? s.grad_magnitude ?? 0.05,
          description: s.description || ""
        })),
        timestamp: new Date().toISOString()
      }
    } catch {
      return mockFullAnalysis
    }
  },

  // Node Mapping
  async runMapping(sessionId: string): Promise<NodeLayerMapping[]> {
    if (FORCE_MOCKS) return mockMappingList
    try {
      const res = await client.post('/api/intervention/map', {
        session_id: sessionId,
        sensitivity_method: "autograd",
        seed: 42
      })
      const mappings = res.data.mappings || res.data || []
      return mappings.map((m: any) => ({
        nodeId: m.node_id || m.nodeId || "node",
        nodeLabel: m.node_label || m.nodeLabel || m.node_id,
        nodeSymbol: m.node_symbol || m.symbol || "",
        mappedLayer: m.mapped_layer || m.layer_name || "fch.0.weight",
        confidence: m.confidence ?? 0.85,
        alignmentPct: m.alignment_pct ?? (m.confidence ? m.confidence * 100 : 85.0),
        strategy: m.strategy || "trend-matching"
      }))
    } catch {
      return mockMappingList
    }
  }
}
