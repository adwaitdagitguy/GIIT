import re
def _prettify(name):
    """Convert snake_case or camelCase node name to readable title."""
    name = re.sub(r'([A-Z])', r' \1', name)
    name = name.replace("_", " ").replace("-", " ")
    return name.strip().title()

def _infer_description(node_name, node_type, domain):
    """
    Infer a meaningful description from node name alone.
    Works for any physics domain.
    """
    name_lower = node_name.lower()

    # Common physics term patterns
    patterns = [
        (["velocity", "vel"],         "Computes the velocity-dependent term in the system dynamics."),
        (["gravity", "grav", "gravitational"], "Computes the gravitational force component acting on the system."),
        (["combiner", "combine", "output", "sum", "add"], "Combines upstream terms to produce the final system output."),
        (["damping", "damp", "drag"], "Models the damping or drag force that dissipates energy in the system."),
        (["spring", "elastic", "restoring"], "Computes the restoring force proportional to displacement."),
        (["input", "source"],         "Receives the primary input signal or driving force for the system."),
        (["potential", "pe"],         "Computes the potential energy component of the system."),
        (["kinetic", "ke"],           "Computes the kinetic energy component of the system."),
        (["force", "thrust"],         "Computes the net force applied to the system."),
        (["acceleration", "accel"],   "Computes the acceleration resulting from applied forces."),
        (["position", "displacement", "pos"], "Tracks the positional state or displacement of the system."),
        (["momentum", "impulse"],     "Computes the momentum or impulse of the system."),
        (["frequency", "freq", "omega"], "Represents the oscillation frequency of the system."),
        (["amplitude", "amp"],        "Represents the amplitude of the system's oscillation."),
        (["temperature", "temp", "thermal"], "Models the thermal state or heat transfer in the system."),
        (["pressure", "press"],       "Computes pressure-related dynamics in the system."),
        (["energy", "power"],         "Tracks energy or power flow through the system."),
        (["hidden", "latent", "layer", "block"], f"Internal neural network block that encodes learned representations for the '{node_name}' component."),
    ]

    for keywords, desc in patterns:
        if any(kw in name_lower for kw in keywords):
            if domain:
                desc = desc + f" (Domain: {domain})"
            return desc

    # Fallback based on node type
    type_descriptions = {
        "input":  f"Input node '{node_name}' — receives external signal or initial condition.",
        "output": f"Output node '{node_name}' — produces the final prediction of the model.",
        "state":  f"State node '{node_name}' — represents an intermediate computed quantity in the model.",
    }
    return type_descriptions.get(node_type, f"Computational node '{node_name}' in the model graph.")

def _infer_term(node_name, formula):
    """Try to find which term in the formula corresponds to this node."""
    if not formula:
        return ""
    name_lower = node_name.lower()

    # Simple heuristic matching
    term_hints = {
        "velocity": ["v₀t", "v0t", "vt", "v·t"],
        "gravity":  ["½gt²", "½g t²", "-½gt²", "0.5gt²"],
        "damping":  ["bẋ", "b·ẋ", "cx'", "damping term"],
        "spring":   ["kx", "k·x", "spring term"],
        "combiner": ["full equation", "entire expression"],
        "output":   ["full output"],
    }
    for key, terms in term_hints.items():
        if key in name_lower:
            return terms[0]
    return ""


def _extract_variables_from_formula(formula, node_name):
    """
    Extract variable symbols from a formula string.
    Returns dict of { symbol: description }
    """
    # Known variable descriptions
    known_vars = {
        "v₀": "Initial velocity",
        "v0": "Initial velocity",
        "v":  "Velocity",
        "t":  "Time",
        "g":  "Gravitational acceleration (9.81 m/s²)",
        "h":  "Height / vertical displacement",
        "x":  "Displacement",
        "k":  "Spring constant",
        "b":  "Damping coefficient",
        "m":  "Mass",
        "ω":  "Angular frequency",
        "ω₀": "Natural frequency",
        "A":  "Amplitude",
        "F":  "Applied force",
        "E":  "Energy",
        "T":  "Temperature / Period",
        "P":  "Pressure / Power",
        "θ":  "Angle",
        "φ":  "Phase angle",
        "ζ":  "Damping ratio",
        "r":  "Radius / distance",
        "c":  "Speed of light / damping constant",
        "a":  "Acceleration",
        "s":  "Displacement / distance",
    }

    # Extract single-character and subscript variables from formula
    found = {}
    # Match patterns like v₀, ω₀, single letters
    symbols = re.findall(r'[a-zA-Zα-ωΑ-Ω][₀-₉]?', formula)
    for sym in symbols:
        if sym in known_vars:
            found[sym] = known_vars[sym]

    return found

def build_node_explanations(graph_data, physics_rules):
    """
    Dynamically builds node explanation dictionary from:
    - graph_data: parsed graph JSON (nodes, edges, metadata)
    - physics_rules: parsed physics rules JSON

    Returns dict of { node_name: { title, description, formula, highlight, variables } }
    """
    explanations = {}

    if graph_data is None:
        return explanations

    raw_nodes = graph_data.get("nodes", [])
    metadata  = graph_data.get("metadata", graph_data.get("model_metadata", {}))
    equations = metadata.get("equations", graph_data.get("residual", {}))
    axis_info = metadata.get("axis_labels", graph_data.get("input_spec", {}))
    domain    = metadata.get("domain", metadata.get("system", ""))
    model_name = metadata.get("model_name", metadata.get("system", "Unknown Model"))

    # Try to get global formula from various possible locations
    global_formula = (
        equations.get("formula", "") or
        metadata.get("equation", "") or
        metadata.get("formula", "") or
        graph_data.get("residual", {}).get("formula", "") or
        ""
    )

    for node in raw_nodes:

        # Support both dict nodes and string nodes
        if isinstance(node, dict):
            node_name   = node.get("id", node.get("name", ""))
            node_type   = node.get("type", "state")
            node_desc   = node.get("description", "")
            node_formula = node.get("formula", "")
            node_vars   = node.get("variables", {})
            node_term   = node.get("highlight", node.get("term", ""))
            node_title  = node.get("title", "")
        else:
            node_name   = str(node)
            node_type   = "state"
            node_desc   = ""
            node_formula = ""
            node_vars   = {}
            node_term   = ""
            node_title  = ""

        if not node_name:
            continue

        # ── Try to get info from physics_rules ──────────────────
        rule = {}
        if physics_rules and isinstance(physics_rules, dict):
            # Support multiple physics_rules formats
            rule = (
                physics_rules.get(node_name, {}) or
                physics_rules.get("nodes", {}).get(node_name, {}) or
                {}
            )
            if isinstance(rule, str):
                # If rule is just a string label, use it as description
                rule = {"description": rule}

        # ── Merge: priority is graph JSON > physics rules > fallback ──
        title       = node_title or rule.get("title", "") or _prettify(node_name)
        description = node_desc  or rule.get("description", "") or _infer_description(node_name, node_type, domain)
        formula     = node_formula or rule.get("formula", "") or global_formula
        highlight   = node_term  or rule.get("highlight", "") or rule.get("term", "") or _infer_term(node_name, formula)
        variables   = node_vars  or rule.get("variables", {}) or {}

        # ── Infer variables from formula if not provided ─────────
        if formula and not variables:
            variables = _extract_variables_from_formula(formula, node_name)

        explanations[node_name] = {
            "title":       title,
            "description": description,
            "formula":     formula if formula else "Not specified in graph metadata",
            "highlight":   highlight if highlight else node_name,
            "variables":   variables,
            "type":        node_type,
            "domain":      domain,
            "model_name":  model_name,
        }

    return explanations