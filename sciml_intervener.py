import torch
import torch.nn as nn
import copy
import json

class TinyMLP(nn.Module):
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


class SciMLIntervener:
    
    def __init__(self, model_path, graph_path):
        
        print("\nINITIALIZATION")
        
        print(f"\nLoading model from: {model_path}")
        self.model = torch.load(model_path, weights_only=False)
        self.model.eval()
        
        num_layers = sum(1 for m in self.model.modules() if isinstance(m, nn.Linear))
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Layers: {num_layers}, Parameters: {num_params}")
        
        print(f"\nLoading graph from: {graph_path}")
        with open(graph_path, 'r') as f:
            self.graph_data = json.load(f)
        
        if 'mapping' not in self.graph_data:
            raise ValueError("Graph JSON must contain 'mapping' field!")
        
        self.mapping = self.graph_data['mapping']
        
        print(f"Loaded {len(self.graph_data['nodes'])} nodes")
        print(f"Loaded {len(self.graph_data['edges'])} edges")
        print(f"\nNode-to-Layer Mapping:")
        for node, layer in self.mapping.items():
            if not hasattr(self.model, layer):
                print(f"WARNING: Layer '{layer}' not found in model!")
            else:
                print(f"  '{node}' -> {layer}")
        
        print("\nBacking up model state...")
        self.original_state = copy.deepcopy(self.model.state_dict())
        print("Original weights saved")
        
        self.original_weights = {}
        self.criterion = nn.MSELoss()
        
        print("\nINITIALIZATION COMPLETE")
    
    
    def get_baseline(self, test_inputs, test_targets):
        
        with torch.no_grad():
            baseline_outputs = self.model(test_inputs)
            baseline_metric = self.criterion(baseline_outputs, test_targets).item()
        
        return baseline_metric
    
    
    def apply_intervention(self, node_name, intervention_type, strength, test_inputs=None):
        
        if node_name not in self.mapping:
            raise ValueError(f"Unknown node: {node_name}. Valid: {list(self.mapping.keys())}")
        
        layer_name = self.mapping[node_name]
        
        if not hasattr(self.model, layer_name):
            raise ValueError(f"Layer '{layer_name}' not found in model!")
        
        self.model.load_state_dict(self.original_state)
        
        layer = getattr(self.model, layer_name)
        
        if intervention_type == "mask":
            with torch.no_grad():
                layer.weight.data *= (1.0 - strength)
                if layer.bias is not None:
                    layer.bias.data *= (1.0 - strength)
        
        elif intervention_type == "perturb":
            print(f"\nAdding noise to {layer_name}")
            
            with torch.no_grad():
                self.original_weights[layer_name] = {
                    'weight': layer.weight.data.clone(),
                    'bias': layer.bias.data.clone() if layer.bias is not None else None
                }
                
                weight_noise = torch.randn_like(layer.weight) * strength
                layer.weight.data += weight_noise
                
                if layer.bias is not None:
                    bias_noise = torch.randn_like(layer.bias) * strength
                    layer.bias.data += bias_noise
                
                print(f"Added noise (mean=0, std={strength})")
        
        elif intervention_type == "freeze":
            print(f"\nFreezing {layer_name} (not fully implemented)")
            pass
        
        else:
            raise ValueError(f"Unknown intervention: {intervention_type}")
        
        if test_inputs is not None:
            with torch.no_grad():
                intervened_outputs = self.model(test_inputs)
            return intervened_outputs
        
        return None
    
    
    def get_plot_data(self, node_name, intervention_type, strength, test_inputs, test_targets):
        
        baseline_metric = self.get_baseline(test_inputs, test_targets)
        
        with torch.no_grad():
            y_preds_baseline = self.model(test_inputs)
        
        y_preds_intervened = self.apply_intervention(
            node_name, 
            intervention_type, 
            strength,
            test_inputs
        )
        
        intervened_metric = self.criterion(y_preds_intervened, test_targets).item()
        
        x_values = test_inputs[:, 0].numpy()
        
        y_preds_baseline_np = y_preds_baseline.squeeze().numpy()
        y_preds_intervened_np = y_preds_intervened.squeeze().numpy()
        y_targets_np = test_targets.squeeze().numpy()
        
        return {
            'x_values': x_values,
            'y_preds_baseline': y_preds_baseline_np,
            'y_preds_intervened': y_preds_intervened_np,
            'y_targets': y_targets_np,
            'baseline_metric': baseline_metric,
            'intervened_metric': intervened_metric,
            'node': node_name,
            'intervention_type': intervention_type,
            'strength': strength
        }
    
    
    def reset_model(self):
        self.model.load_state_dict(self.original_state)
        self.original_weights = {}


if __name__ == "__main__":
    print("\nPHASE 2 TEST")
    
    intervener = SciMLIntervener(
        model_path="tiny_mlp.pt",
        graph_path="graph_phase2.json"
    )
    
    test_inputs = torch.load("test_inputs.pt")
    test_targets = torch.load("test_targets.pt")
    
    print("\nTEST 1: Mask Intervention")
    
    plot_data = intervener.get_plot_data(
        node_name="input_processor",
        intervention_type="mask",
        strength=1.0,
        test_inputs=test_inputs,
        test_targets=test_targets
    )
    
    print(f"\nBaseline MSE: {plot_data['baseline_metric']:.4f}")
    print(f"Intervened MSE: {plot_data['intervened_metric']:.4f}")
    print(f"Change: {plot_data['intervened_metric'] - plot_data['baseline_metric']:+.4f}")
    
    intervener.reset_model()
    
    print("\nTEST 2: Perturb Intervention")
    
    plot_data2 = intervener.get_plot_data(
        node_name="input_processor",
        intervention_type="perturb",
        strength=0.5,
        test_inputs=test_inputs,
        test_targets=test_targets
    )
    
    print(f"\nBaseline MSE: {plot_data2['baseline_metric']:.4f}")
    print(f"Intervened MSE: {plot_data2['intervened_metric']:.4f}")
    print(f"Change: {plot_data2['intervened_metric'] - plot_data2['baseline_metric']:+.4f}")
    
    print("\nPHASE 2 TASKS COMPLETE")