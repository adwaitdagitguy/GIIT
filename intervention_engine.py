"""
Tasks 2-6: Intervention Engine
Implements: hardcoded mapping, baseline evaluation, interventions, results return

Following major_tasks.pdf requirements exactly
"""

import torch
import torch.nn as nn
import copy

class TinyMLP(nn.Module):
    """Model architecture (must match create_and_train_model.py)"""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


class InterventionEngine:
    """
    TASK 2: Hardcode Mapping
    TASK 3: Baseline Evaluation  
    TASK 4: Apply Intervention (mask only)
    TASK 5: Rerun Evaluation
    TASK 6: Return Results
    """
    
    def __init__(self, model_path, test_inputs_path, test_targets_path):
        """
        Initialize the intervention engine
        
        Args:
            model_path: Path to saved .pt model
            test_inputs_path: Path to test inputs
            test_targets_path: Path to test targets
        """
        
        # Load model
        print(f"\nLoading model from: {model_path}")
        self.model = torch.load(model_path, weights_only=False)
        self.model.eval()
        
        # Count parameters (Task 1 requirement)
        num_layers = sum(1 for m in self.model.modules() if isinstance(m, nn.Linear))
        num_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"Number of layers: {num_layers}")
        print(f"Number of parameters: {num_params}")
        
        # Load test data
        print(f"\nLoading test data...")
        self.test_inputs = torch.load(test_inputs_path)
        self.test_targets = torch.load(test_targets_path)
        print(f"Loaded {len(self.test_inputs)} test cases")
        
        # TASK 2: Hardcoded Mapping
        # This maps conceptual "nodes" to actual layer names
        self.mapping = {
            "input_processor": "layer1",
            "output_computer": "layer2"
        }
        
        print(f"\nNode-to-Layer Mapping:")
        for node, layer in self.mapping.items():
            print(f"  '{node}' → {layer}")
        
        # Store original model state
        self.original_state = copy.deepcopy(self.model.state_dict())
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        print("INITIALIZATION COMPLETE")
    
    
    def get_baseline(self):
        """
        TASK 3: Baseline Evaluation
        
        Returns:
            float: Baseline MSE metric
        """
        # print("COMPUTING BASELINE METRIC")
        
        with torch.no_grad():
            baseline_outputs = self.model(self.test_inputs)
            baseline_metric = self.criterion(baseline_outputs, self.test_targets).item()
        
        # print(f"Baseline MSE: {baseline_metric:.6f}")
        
        return baseline_metric
    
    
    def apply_intervention(self, node_name, intervention_type="mask", strength=1.0):
        
        # print("APPLYING INTERVENTION")
        
        # Validate node name
        if node_name not in self.mapping:
            raise ValueError(f"Unknown node: {node_name}. Valid nodes: {list(self.mapping.keys())}")
        
        # Get layer name from mapping
        layer_name = self.mapping[node_name]
        # print(f"Node: '{node_name}' → Layer: '{layer_name}'")
        # print(f"Intervention: {intervention_type}")
        # print(f"Strength: {strength:.2f}")
        
        # Reset model to original state first
        self.model.load_state_dict(self.original_state)
        
        # Get the layer
        layer = getattr(self.model, layer_name)
        
        # TASK 4: Implement MASK intervention only (Phase 1)
        if intervention_type == "mask":
            # print(f"\nMasking {layer_name}...")
            
            with torch.no_grad():
                # Zero out weights proportional to strength
                # strength=1.0 means complete masking (all zeros)
                # strength=0.5 means 50% masking
                # strength=0.0 means no masking
                layer.weight.data *= (1.0 - strength)
                
                if layer.bias is not None:
                    layer.bias.data *= (1.0 - strength)
            
            # print(f"Applied mask with strength {strength:.2f}")
            
        elif intervention_type == "freeze":
            # Phase 2 feature
            # print("'freeze' intervention not implemented in Phase 1")
            # print("Using 'mask' instead")
            self.apply_intervention(node_name, "mask", strength)
            
        elif intervention_type == "perturb":
            # Phase 2 feature  
            # print("'perturb' intervention not implemented in Phase 1")
            # print("Using 'mask' instead")
            self.apply_intervention(node_name, "mask", strength)
        
        else:
            raise ValueError(f"Unknown intervention type: {intervention_type}")
        
        # TASK 5: Rerun Evaluation
        # print("EVALUATING INTERVENED MODEL")
        
        with torch.no_grad():
            intervened_outputs = self.model(self.test_inputs)
            intervened_metric = self.criterion(intervened_outputs, self.test_targets).item()
        
        # print(f"Intervened MSE: {intervened_metric:.6f}")
        
        return intervened_metric
    
    
    def run_intervention_experiment(self, node_name, intervention_type="mask", strength=1.0):
        # print("FULL INTERVENTION EXPERIMENT")
        
        # Get baseline
        baseline_metric = self.get_baseline()
        
        # Apply intervention and get new metric
        intervened_metric = self.apply_intervention(node_name, intervention_type, strength)
        
        # Calculate changes
        delta = intervened_metric - baseline_metric
        percent_change = (delta / baseline_metric * 100) if baseline_metric > 0 else 0
        
        # TASK 6: Return results
        results = {
            "baseline_metric": baseline_metric,
            "intervened_metric": intervened_metric,
            "delta": delta,
            "percent_change": percent_change,
            "node": node_name,
            "layer": self.mapping[node_name],
            "intervention_type": intervention_type,
            "strength": strength
        }
        
        # Print summary
        print("RESULTS SUMMARY")
        print(f"Node intervened: {node_name}")
        print(f"Layer affected:  {self.mapping[node_name]}")
        print(f"Intervention:    {intervention_type} (strength={strength:.2f})")
        print(f"\nBaseline MSE:    {baseline_metric:.6f}")
        print(f"Intervened MSE:  {intervened_metric:.6f}")
        print(f"Change:          {delta:+.6f} ({percent_change:+.2f}%)")

        # Interpretation
        if abs(percent_change) > 100:
            print("SEVERE IMPACT: Model heavily relies on this component")
        elif abs(percent_change) > 10:
            print("MODERATE IMPACT: Model partially depends on this component")
        else:
            print("MINIMAL IMPACT: Model doesn't rely much on this component")
        
        
        return results
    
    
    def reset_model(self):
        self.model.load_state_dict(self.original_state)
        print("Model reset to original state")
    
    
    def get_sample_predictions(self, n_samples=3):
       
        n = min(n_samples, len(self.test_inputs))
        
        with torch.no_grad():
            predictions = self.model(self.test_inputs[:n])
        
        return {
            "inputs": self.test_inputs[:n],
            "targets": self.test_targets[:n],
            "predictions": predictions
        }


if __name__ == "__main__":

    # Initialize engine
    engine = InterventionEngine(
        model_path="tiny_mlp.pt",
        test_inputs_path="test_inputs.pt",
        test_targets_path="test_targets.pt"
    )
    
    print("\n# TEST 1: Intervene on 'input_processor' node")
    results1 = engine.run_intervention_experiment(
        node_name="input_processor",
        intervention_type="mask",
        strength=1.0
    )
    
    engine.reset_model()
    
    print("\n# TEST 2: Intervene on 'output_computer' node")
    results2 = engine.run_intervention_experiment(
        node_name="output_computer",
        intervention_type="mask",
        strength=1.0
    )
    engine.reset_model()
    
   
    print("\n# TEST 3: Partial intervention (50% strength)")
    results3 = engine.run_intervention_experiment(
        node_name="input_processor",
        intervention_type="mask",
        strength=0.5
    )
    
    # Final summary
    print("# COMPARISON OF ALL TESTS")
    print(f"\n{'Node':<20} {'Strength':<10} {'Impact':<15}")
    print(f"{'input_processor':<20} {1.0:<10.1f} {results1['percent_change']:+.2f}%")
    print(f"{'output_computer':<20} {1.0:<10.1f} {results2['percent_change']:+.2f}%")
    print(f"{'input_processor':<20} {0.5:<10.1f} {results3['percent_change']:+.2f}%")

    
