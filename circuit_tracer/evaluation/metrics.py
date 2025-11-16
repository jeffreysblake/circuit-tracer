"""Metrics for evaluating circuit quality and task alignment."""

import logging
from typing import Any

import torch
import torch.nn.functional as F
from circuit_tracer.graph import Graph
from circuit_tracer.replacement_model import ReplacementModel

logger = logging.getLogger(__name__)


def compute_feature_coverage(
    graph: Graph, expected_features: list[dict[str, Any]] | None = None
) -> float:
    """Compute what percentage of expected feature patterns are present in the graph.

    Args:
        graph: Attribution graph to analyze
        expected_features: List of feature patterns to look for. Each pattern has:
            - type: "layer_position" or "pattern" or "semantic"
            - name: identifier for the feature
            - description: what the feature represents
            - matcher: optional dict with matching criteria

    Returns:
        Coverage score between 0.0 and 1.0
    """
    if not expected_features:
        logger.warning("No expected features provided, returning 1.0")
        return 1.0

    # Count how many features are active in the graph
    n_active = len(graph.active_features)

    # For now, we use a simple heuristic: higher feature count suggests better coverage
    # In a real implementation, you'd match against specific feature patterns
    # This is a placeholder that can be enhanced with feature description matching

    if n_active == 0:
        return 0.0

    # Simple heuristic: normalize by expected number
    # This should be replaced with actual pattern matching
    expected_count = len(expected_features)
    coverage = min(1.0, n_active / max(expected_count * 10, 1))

    return coverage


def compute_pathway_coherence(graph: Graph) -> float:
    """Compute how coherent and interpretable the circuit pathways are.

    Coherence is measured by:
    - Path lengths between features (shorter = more direct)
    - Edge weight distribution (concentrated = clearer signal)
    - Graph connectivity structure

    Args:
        graph: Attribution graph to analyze

    Returns:
        Coherence score between 0.0 and 1.0
    """
    adj_matrix = graph.adjacency_matrix.abs()

    if adj_matrix.numel() == 0:
        return 0.0

    # Measure 1: Edge weight concentration (higher Gini = more concentrated)
    # Sort edge weights in descending order
    sorted_weights, _ = torch.sort(adj_matrix.flatten(), descending=True)
    nonzero_weights = sorted_weights[sorted_weights > 1e-8]

    if len(nonzero_weights) == 0:
        return 0.0

    # Compute concentration: what fraction of total influence comes from top edges?
    total_influence = nonzero_weights.sum()
    top_10_percent = max(1, len(nonzero_weights) // 10)
    top_influence = nonzero_weights[:top_10_percent].sum()
    concentration = (top_influence / total_influence).item()

    # Measure 2: Average path efficiency
    # Higher average edge weight suggests more direct connections
    mean_edge_weight = nonzero_weights.mean().item()
    max_edge_weight = nonzero_weights.max().item()
    efficiency = mean_edge_weight / (max_edge_weight + 1e-8)

    # Combine measures
    coherence = 0.6 * concentration + 0.4 * efficiency

    return float(coherence)


def compute_sparsity(graph: Graph) -> float:
    """Compute how sparse the attribution graph is.

    Sparsity is desirable because it indicates the model uses a small,
    interpretable set of features rather than diffuse activations.

    Args:
        graph: Attribution graph to analyze

    Returns:
        Sparsity score between 0.0 (dense) and 1.0 (very sparse)
    """
    adj_matrix = graph.adjacency_matrix

    # Count non-zero edges
    total_edges = adj_matrix.numel()
    nonzero_edges = (adj_matrix.abs() > 1e-8).sum().item()

    if total_edges == 0:
        return 1.0

    # Sparsity = 1 - (fraction of non-zero edges)
    sparsity = 1.0 - (nonzero_edges / total_edges)

    return float(sparsity)


def compute_graph_density(graph: Graph) -> float:
    """Compute the density of the attribution graph.

    Density = (number of edges) / (number of possible edges)

    Args:
        graph: Attribution graph to analyze

    Returns:
        Density score between 0.0 (no edges) and 1.0 (fully connected)
    """
    adj_matrix = graph.adjacency_matrix

    n_nodes = adj_matrix.shape[0]
    if n_nodes <= 1:
        return 0.0

    # Maximum possible edges in a directed graph
    max_edges = n_nodes * (n_nodes - 1)

    # Count actual edges (above threshold)
    actual_edges = (adj_matrix.abs() > 1e-8).sum().item()

    density = actual_edges / max_edges if max_edges > 0 else 0.0

    return float(density)


def compute_intervention_stability(
    graph: Graph, model: ReplacementModel, interventions: list[dict[str, Any]] | None = None
) -> float:
    """Compute how stable the circuit is under feature interventions.

    This tests whether manipulating key features has predictable effects.

    Args:
        graph: Attribution graph to analyze
        model: ReplacementModel to run interventions on
        interventions: List of intervention specifications

    Returns:
        Stability score between 0.0 (unstable) and 1.0 (very stable)

    Note:
        This is a placeholder implementation. Real stability testing requires
        running the model with interventions and measuring output changes.
    """
    if not interventions:
        logger.warning("No interventions specified, returning 0.5 (neutral)")
        return 0.5

    # Placeholder: In a real implementation, you would:
    # 1. Select top features from the graph
    # 2. Run interventions (ablation, amplification, etc.)
    # 3. Measure output change consistency
    # 4. Return score based on predictability

    # For now, return a neutral score
    return 0.5


def compute_top_k_feature_influence(graph: Graph, k: int = 10) -> dict[str, Any]:
    """Identify the top-k most influential features in the graph.

    Args:
        graph: Attribution graph to analyze
        k: Number of top features to return

    Returns:
        Dictionary containing:
            - feature_indices: Tensor of top-k feature indices
            - influence_scores: Influence score for each feature
            - layer_distribution: Count of features per layer
    """
    adj_matrix = graph.adjacency_matrix.abs()

    # Compute influence as sum of outgoing edge weights
    # (how much this feature affects others)
    n_features = len(graph.active_features)

    if n_features == 0:
        return {
            "feature_indices": torch.tensor([]),
            "influence_scores": torch.tensor([]),
            "layer_distribution": {},
        }

    # Get feature portion of adjacency matrix (exclude error, embed, logit nodes)
    feature_adj = adj_matrix[:n_features, :n_features]

    # Sum outgoing edges for each feature
    outgoing_influence = feature_adj.sum(dim=0)

    # Get top-k
    actual_k = min(k, n_features)
    top_scores, top_indices = torch.topk(outgoing_influence, actual_k)

    # Get layer distribution
    layer_distribution = {}
    for idx in top_indices:
        layer = graph.active_features[idx][0].item()
        layer_distribution[layer] = layer_distribution.get(layer, 0) + 1

    return {
        "feature_indices": top_indices,
        "influence_scores": top_scores,
        "layer_distribution": layer_distribution,
    }
