import { ParsedGraph, ModelMetadata, InterventionResult, SweepResult, SensitivityItem, NodeLayerMapping, FullAnalysisResult } from '../types/api'

// Verified Damped Harmonic Oscillator Graph Specification
export const mockParsedGraph: ParsedGraph = {
  metadata: {
    system: "Damped Harmonic Oscillator",
    equation: "y'' + \\mu y' + k y = 0",
    description: "Underdamped physical harmonic oscillator with linear viscous damping and Hookean spring restoring force.",
    pinn_input: "t (time, [0, 1])",
    pinn_output: "y (displacement)",
    axis_labels: {
      x_label: "Time (t)",
      x_unit: "s",
      y_label: "Displacement (y)",
      y_unit: "m"
    },
    domain: "0 <= t <= 1.0",
    model_name: "FCN_Oscillator"
  },
  nodes: [
    {
      id: "node_t",
      label: "Time Input",
      symbol: "t",
      type: "input",
      unit: "s",
      description: "Temporal coordinate input to the PINN neural network.",
      formula: "t \\in [0, 1]"
    },
    {
      id: "node_latent_1",
      label: "Latent Features (fcs.0)",
      symbol: "h_1",
      type: "latent",
      unit: "dimensionless",
      description: "First hidden representation layer transforming temporal input to high-dimensional feature space.",
      formula: "\\sigma(W_1 t + b_1)"
    },
    {
      id: "node_mu",
      label: "Damping Coefficient",
      symbol: "\\mu",
      type: "parameter",
      unit: "N·s/m",
      default_value: 4.0,
      expected_sensitivity_sign: "-",
      trend: "Higher damping causes exponential amplitude decay rate to increase.",
      description: "Viscous resistance factor dissipating mechanical energy in the oscillator.",
      formula: "F_{damp} = -\\mu \\dot{y}"
    },
    {
      id: "node_k",
      label: "Spring Stiffness",
      symbol: "k",
      type: "parameter",
      unit: "N/m",
      default_value: 400.0,
      expected_sensitivity_sign: "+",
      trend: "Higher stiffness increases oscillation frequency (\\omega_0 = \\sqrt{k}).",
      description: "Restoring force constant governing natural frequency \\omega_0.",
      formula: "F_{spring} = -k y"
    },
    {
      id: "node_v",
      label: "Velocity",
      symbol: "\\dot{y}",
      type: "intermediate",
      unit: "m/s",
      description: "First time-derivative of displacement \\partial y / \\partial t computed via autograd.",
      formula: "\\dot{y} = \\frac{dy}{dt}"
    },
    {
      id: "node_a",
      label: "Acceleration",
      symbol: "\\ddot{y}",
      type: "intermediate",
      unit: "m/s^2",
      description: "Second time-derivative of displacement \\partial^2 y / \\partial t^2 computed via autograd.",
      formula: "\\ddot{y} = \\frac{d^2y}{dt^2}"
    },
    {
      id: "node_residual",
      label: "Physics Loss Residual",
      symbol: "f_{ode}",
      type: "physics_loss",
      unit: "N",
      description: "Governing ODE residual loss evaluated at collocation points.",
      formula: "\\ddot{y} + \\mu \\dot{y} + k y = 0"
    },
    {
      id: "node_y",
      label: "Displacement Output",
      symbol: "y(t)",
      type: "output",
      unit: "m",
      description: "Predicted oscillator position trajectory over time.",
      formula: "y(t) = e^{-\\frac{\\mu}{2} t} \\cos(\\omega_d t)"
    }
  ],
  edges: [
    { source: "node_t", target: "node_latent_1", label: "forward" },
    { source: "node_latent_1", target: "node_y", label: "decode" },
    { source: "node_y", target: "node_v", label: "d/dt" },
    { source: "node_v", target: "node_a", label: "d/dt" },
    { source: "node_mu", target: "node_residual", label: "\\times \\dot{y}" },
    { source: "node_k", target: "node_residual", label: "\\times y" },
    { source: "node_a", target: "node_residual", label: "+" },
    { source: "node_v", target: "node_residual", label: "+" },
    { source: "node_y", target: "node_residual", label: "+" }
  ]
}

export const mockModelMetadata: ModelMetadata = {
  sessionId: "session_damped_oscillator_demo",
  modelClass: "FCN (Fully Connected PINN)",
  totalParameters: 2241,
  layerNames: ["fcs.0.weight", "fcs.0.bias", "fch.0.weight", "fch.0.bias", "fch.1.weight", "fch.1.bias", "fce.0.weight", "fce.0.bias"],
  perturbableTensors: ["fcs.0.weight", "fch.0.weight", "fch.1.weight", "fce.0.weight"],
  loadedAt: new Date().toISOString()
}

// Analytical harmonic oscillator trajectory generator
export function generateOscillatorTrajectory(mu = 4.0, k = 400.0, perturbation = 0.0, numPoints = 120) {
  const tVals: number[] = []
  const yBase: number[] = []
  const yInter: number[] = []
  const yExact: number[] = []

  const omega0 = Math.sqrt(k)
  const gamma = mu / 2.0
  const omegaD = Math.sqrt(Math.max(0.1, omega0 * omega0 - gamma * gamma))

  // Perturbed parameters based on intervention strength
  const gammaPerturbed = gamma * (1.0 + perturbation * 0.8)
  const omegaDPerturbed = omegaD * (1.0 - perturbation * 0.35)

  for (let i = 0; i < numPoints; i++) {
    const t = i / (numPoints - 1)
    tVals.push(t)

    // Baseline PINN prediction
    const base = Math.exp(-gamma * t) * Math.cos(omegaD * t)
    yBase.push(base)
    yExact.push(base)

    // Intervened trajectory
    const noise = (Math.sin(t * 35.0) * 0.08 + Math.cos(t * 18.0) * 0.05) * perturbation
    const inter = Math.exp(-gammaPerturbed * t) * Math.cos(omegaDPerturbed * t) + noise
    yInter.push(inter)
  }

  return { tVals, yBase, yInter, yExact }
}

export function createMockInterventionResult(
  layerName = "fch.0.weight",
  interventionType: "noise" | "masking" = "noise",
  strength = 0.4
): InterventionResult {
  const { tVals, yBase, yInter, yExact } = generateOscillatorTrajectory(4.0, 400.0, strength)

  let sumDiffSq = 0
  let sumAbsDiff = 0
  let maxDiv = 0
  let sumBaseSq = 0
  let sumInterSq = 0

  for (let i = 0; i < tVals.length; i++) {
    const diff = yInter[i] - yBase[i]
    sumDiffSq += diff * diff
    sumAbsDiff += Math.abs(diff)
    if (Math.abs(diff) > maxDiv) maxDiv = Math.abs(diff)
    sumBaseSq += yBase[i] * yBase[i]
    sumInterSq += yInter[i] * yInter[i]
  }

  const n = tVals.length
  const mseIntervened = sumDiffSq / n
  const l2Baseline = Math.sqrt(sumBaseSq)
  const l2Intervened = Math.sqrt(sumInterSq)

  return {
    layerName,
    interventionType,
    strength,
    timestamp: new Date().toISOString(),
    trajectory: {
      xValues: tVals,
      yBaseline: yBase,
      yIntervened: yInter,
      yGroundTruth: yExact
    },
    metrics: {
      mseBaseline: 0.00012,
      mseIntervened: parseFloat(mseIntervened.toFixed(6)),
      l2Baseline: parseFloat(l2Baseline.toFixed(4)),
      l2Intervened: parseFloat(l2Intervened.toFixed(4)),
      rmse: parseFloat(Math.sqrt(mseIntervened).toFixed(6)),
      mae: parseFloat((sumAbsDiff / n).toFixed(6)),
      maxDivergence: parseFloat(maxDiv.toFixed(4)),
      percentDegradation: parseFloat(((mseIntervened / 0.00012 - 1) * 100).toFixed(1))
    }
  }
}

export function createMockSweepResult(
  layerName = "fch.0.weight",
  interventionType: "noise" | "masking" = "noise"
): SweepResult {
  const strengths = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  const mseValues: number[] = []
  const l2Norms: number[] = []
  const maxDivergences: number[] = []

  strengths.forEach((s) => {
    const res = createMockInterventionResult(layerName, interventionType, s)
    mseValues.push(res.metrics.mseIntervened)
    l2Norms.push(res.metrics.l2Intervened)
    maxDivergences.push(res.metrics.maxDivergence)
  })

  return {
    layerName,
    interventionType,
    strengths,
    mseValues,
    l2Norms,
    maxDivergences
  }
}

export const mockSensitivityList: SensitivityItem[] = [
  {
    parameter: "mu",
    label: "Damping (\\mu)",
    gradSign: "-",
    expectedSign: "-",
    consistent: true,
    magnitude: 0.0482,
    description: "Autograd gradient aligns with dissipative damping physics."
  },
  {
    parameter: "k",
    label: "Stiffness (k)",
    gradSign: "+",
    expectedSign: "+",
    consistent: true,
    magnitude: 0.1845,
    description: "Autograd gradient confirms higher frequency response with increased stiffness."
  },
  {
    parameter: "initial_velocity",
    label: "Init Velocity (v_0)",
    gradSign: "+",
    expectedSign: "+",
    consistent: true,
    magnitude: 0.0912,
    description: "Initial kinetic energy gradient matches forward trajectory."
  }
]

export const mockMappingList: NodeLayerMapping[] = [
  {
    nodeId: "node_mu",
    nodeLabel: "Damping Coefficient (\\mu)",
    nodeSymbol: "\\mu",
    mappedLayer: "fch.0.weight",
    confidence: 0.94,
    alignmentPct: 94.2,
    strategy: "trend-matching"
  },
  {
    nodeId: "node_k",
    nodeLabel: "Spring Stiffness (k)",
    nodeSymbol: "k",
    mappedLayer: "fch.1.weight",
    confidence: 0.89,
    alignmentPct: 88.7,
    strategy: "gradient-sensitivity"
  },
  {
    nodeId: "node_v",
    nodeLabel: "Velocity Derivative (\\dot{y})",
    nodeSymbol: "\\dot{y}",
    mappedLayer: "fcs.0.weight",
    confidence: 0.82,
    alignmentPct: 81.5,
    strategy: "direct-coupling"
  },
  {
    nodeId: "node_y",
    nodeLabel: "Displacement Output (y)",
    nodeSymbol: "y",
    mappedLayer: "fce.0.weight",
    confidence: 0.98,
    alignmentPct: 98.1,
    strategy: "trend-matching"
  }
]

export const mockFullAnalysis: FullAnalysisResult = {
  l2NormBaseline: 7.8421,
  physicsResidualNorm: 0.00318,
  sensitivity: mockSensitivityList,
  timestamp: new Date().toISOString()
}
