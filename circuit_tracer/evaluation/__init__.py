"""Evaluation framework for task-specific circuit analysis."""

from circuit_tracer.evaluation.task_evaluator import TaskEvaluator, EvaluationResult
from circuit_tracer.evaluation.task_config import TaskConfig, load_task_config
from circuit_tracer.evaluation.metrics import (
    compute_feature_coverage,
    compute_pathway_coherence,
    compute_sparsity,
    compute_graph_density,
)

__all__ = [
    "TaskEvaluator",
    "EvaluationResult",
    "TaskConfig",
    "load_task_config",
    "compute_feature_coverage",
    "compute_pathway_coherence",
    "compute_sparsity",
    "compute_graph_density",
]
