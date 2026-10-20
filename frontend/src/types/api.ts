export type NodeType = "input" | "parameter" | "intermediate" | "output" | "physics_loss" | "latent" | "state"

export interface GraphNode {
  id: string
  label: string
  symbol: string
  type: NodeType
  unit?: string
  description?: string
  formula?: string
  default_value?: number
  expected_sensitivity_sign?: "+" | "-"
  trend?: string
}

export interface GraphEdge {
  source: string
  target: string
  label?: string
}

export interface InputSpec {
  names: string[]
  dim: number
  ranges?: Record<string, [number, number]>
}

export interface OutputSpec {
  names: string[]
  dim: number
}

export interface ParameterSpec {
  label: string
  default_value: number
  expected_sensitivity_sign: "+" | "-"
  trend: string
}

export interface ResidualSpec {
  formula: string
  description: string
  derivatives?: Record<string, any>
}

export interface ParsedGraph {
  metadata: {
    system: string
    equation: string
    description: string
    pinn_input?: string
    pinn_output?: string
    axis_labels?: {
      x_label: string
      x_unit: string
      y_label: string
      y_unit: string
    }
    domain?: string
    model_name?: string
  }
  nodes: GraphNode[]
  edges: GraphEdge[]
  input_spec?: InputSpec
  output_spec?: OutputSpec
  parameters?: Record<string, ParameterSpec>
  residual?: ResidualSpec
  expected_trends?: Record<string, string>
  test_data_generator?: string
}

export interface ModelMetadata {
  sessionId: string
  modelClass: string
  totalParameters: number
  layerNames: string[]
  perturbableTensors: string[]
  loadedAt: string
}

export type InterventionType = "noise" | "masking" | "ablate" | "clamp" | "gain"

export interface InterventionConfig {
  sessionId: string
  layerName: string
  interventionType: InterventionType
  strength: number
  inputRange?: [number, number]
}

export interface TrajectoryData {
  xValues: number[]
  yBaseline: number[]
  yIntervened: number[]
  yGroundTruth?: number[]
}

export interface ErrorMetrics {
  mseBaseline: number
  mseIntervened: number
  l2Baseline: number
  l2Intervened: number
  rmse: number
  mae: number
  maxDivergence: number
  percentDegradation: number
}

export interface InterventionResult {
  trajectory: TrajectoryData
  metrics: ErrorMetrics
  layerName: string
  interventionType: InterventionType
  strength: number
  timestamp: string
}

export interface SweepResult {
  layerName: string
  interventionType: InterventionType
  strengths: number[]
  mseValues: number[]
  l2Norms: number[]
  maxDivergences: number[]
}

export interface SensitivityItem {
  parameter: string
  label: string
  gradSign: "+" | "-" | "0"
  expectedSign: "+" | "-"
  consistent: boolean
  magnitude: number
  description?: string
}

export interface NodeLayerMapping {
  nodeId: string
  nodeLabel: string
  nodeSymbol: string
  mappedLayer: string
  confidence: number
  alignmentPct: number
  strategy: "trend-matching" | "gradient-sensitivity" | "direct-coupling"
}

export interface FullAnalysisResult {
  l2NormBaseline: number
  physicsResidualNorm: number
  sensitivity: SensitivityItem[]
  timestamp: string
}

export interface AssistantMessage {
  id: string
  role: "user" | "assistant" | "system"
  content: string
  timestamp: string
}
