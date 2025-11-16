"""Task-specific evaluation of attribution graphs."""

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from tqdm import tqdm

from circuit_tracer.attribution.attribute import attribute
from circuit_tracer.graph import Graph
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.evaluation.metrics import (
    compute_feature_coverage,
    compute_pathway_coherence,
    compute_sparsity,
    compute_graph_density,
    compute_intervention_stability,
    compute_top_k_feature_influence,
)
from circuit_tracer.evaluation.task_config import TaskConfig

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Results from evaluating a single prompt."""

    prompt: str
    graph: Graph
    metrics: dict[str, float] = field(default_factory=dict)
    task_features: dict[str, Any] = field(default_factory=dict)
    interpretation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Get a summary of the evaluation result (without full graph)."""
        return {
            "prompt": self.prompt[:100] + "..." if len(self.prompt) > 100 else self.prompt,
            "metrics": self.metrics,
            "n_active_features": len(self.graph.active_features),
            "n_logits": len(self.graph.logit_tokens),
            "task_features": self.task_features,
            "interpretation": self.interpretation,
            "metadata": self.metadata,
        }


@dataclass
class BatchEvaluationResults:
    """Results from evaluating a batch of prompts."""

    task_name: str
    model_name: str
    results: list[EvaluationResult] = field(default_factory=list)
    aggregate_metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Get a summary of all results."""
        return {
            "task_name": self.task_name,
            "model_name": self.model_name,
            "n_prompts": len(self.results),
            "aggregate_metrics": self.aggregate_metrics,
            "individual_results": [r.summary() for r in self.results],
            "metadata": self.metadata,
        }


class TaskEvaluator:
    """Evaluate how well a model's circuits align with a specific task."""

    def __init__(
        self,
        model: ReplacementModel,
        task_config: TaskConfig,
        attribution_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize the TaskEvaluator.

        Args:
            model: ReplacementModel to evaluate
            task_config: Configuration for the task
            attribution_kwargs: Additional kwargs for the attribute() function
        """
        self.model = model
        self.task_config = task_config
        self.attribution_kwargs = attribution_kwargs or {}

        # Set defaults for attribution if not specified
        if "max_n_logits" not in self.attribution_kwargs:
            self.attribution_kwargs["max_n_logits"] = 10
        if "batch_size" not in self.attribution_kwargs:
            self.attribution_kwargs["batch_size"] = 256

    def evaluate_prompt(
        self, prompt: str, verbose: bool = False
    ) -> EvaluationResult:
        """Evaluate a single prompt.

        Args:
            prompt: The input prompt to evaluate
            verbose: Whether to show progress information

        Returns:
            EvaluationResult containing graph and metrics
        """
        # Run attribution
        logger.info(f"Running attribution on prompt: {prompt[:50]}...")
        graph = attribute(
            prompt=prompt,
            model=self.model,
            verbose=verbose,
            **self.attribution_kwargs,
        )

        # Compute metrics
        metrics = self._compute_metrics(graph)

        # Identify task-specific features
        task_features = self._identify_task_features(graph)

        # Generate interpretation
        interpretation = self._generate_interpretation(graph, metrics, task_features)

        return EvaluationResult(
            prompt=prompt,
            graph=graph,
            metrics=metrics,
            task_features=task_features,
            interpretation=interpretation,
            metadata={
                "task_name": self.task_config.task_name,
                "model_cfg": str(self.model.cfg.model_name),
            },
        )

    def evaluate_batch(
        self,
        prompts: list[str],
        verbose: bool = True,
        save_graphs: bool = False,
    ) -> BatchEvaluationResults:
        """Evaluate a batch of prompts.

        Args:
            prompts: List of prompts to evaluate
            verbose: Whether to show progress bar
            save_graphs: Whether to keep full graphs in results (uses memory)

        Returns:
            BatchEvaluationResults with individual and aggregate metrics
        """
        results = []

        iterator = tqdm(prompts, desc="Evaluating prompts") if verbose else prompts

        for prompt in iterator:
            result = self.evaluate_prompt(prompt, verbose=False)

            # Optionally clear graph to save memory
            if not save_graphs:
                # Keep graph metadata but clear large tensors
                result.metadata["graph_summary"] = {
                    "n_features": len(result.graph.active_features),
                    "n_tokens": len(result.graph.input_tokens),
                    "n_logits": len(result.graph.logit_tokens),
                }
                result.graph = None  # Clear to save memory

            results.append(result)

        # Compute aggregate metrics
        aggregate_metrics = self._compute_aggregate_metrics(results)

        return BatchEvaluationResults(
            task_name=self.task_config.task_name,
            model_name=str(self.model.cfg.model_name),
            results=results,
            aggregate_metrics=aggregate_metrics,
            metadata={
                "n_prompts": len(prompts),
                "task_description": self.task_config.description,
            },
        )

    def _compute_metrics(self, graph: Graph) -> dict[str, float]:
        """Compute all enabled metrics for a graph.

        Args:
            graph: Attribution graph to analyze

        Returns:
            Dictionary of metric names to scores
        """
        metrics = {}

        # Get metric configurations
        metric_configs = self.task_config.evaluation_metrics

        # Compute each enabled metric
        if metric_configs.get("coverage", MetricConfig()).enabled:
            metrics["coverage"] = compute_feature_coverage(
                graph, [feat.__dict__ for feat in self.task_config.expected_features]
            )

        if metric_configs.get("coherence", MetricConfig()).enabled:
            metrics["coherence"] = compute_pathway_coherence(graph)

        if metric_configs.get("sparsity", MetricConfig()).enabled:
            metrics["sparsity"] = compute_sparsity(graph)

        if metric_configs.get("density", MetricConfig()).enabled:
            metrics["density"] = compute_graph_density(graph)

        # Stability requires interventions, which is expensive
        # Only compute if explicitly enabled
        if metric_configs.get("stability", MetricConfig(enabled=False)).enabled:
            metrics["stability"] = compute_intervention_stability(
                graph, self.model, [t.__dict__ for t in self.task_config.intervention_tests]
            )

        # Compute weighted score
        total_weight = sum(
            cfg.weight for cfg in metric_configs.values() if cfg.enabled
        )
        if total_weight > 0:
            weighted_score = sum(
                metrics.get(name, 0) * cfg.weight
                for name, cfg in metric_configs.items()
                if cfg.enabled
            ) / total_weight
            metrics["weighted_score"] = weighted_score

        return metrics

    def _identify_task_features(self, graph: Graph) -> dict[str, Any]:
        """Identify features relevant to the task.

        Args:
            graph: Attribution graph to analyze

        Returns:
            Dictionary with task-specific feature information
        """
        # Get top influential features
        top_features = compute_top_k_feature_influence(graph, k=10)

        return {
            "top_influential_features": top_features,
            "n_active_features": len(graph.active_features),
            "layer_distribution": top_features.get("layer_distribution", {}),
        }

    def _generate_interpretation(
        self,
        graph: Graph,
        metrics: dict[str, float],
        task_features: dict[str, Any],
    ) -> str:
        """Generate a human-readable interpretation of the results.

        Args:
            graph: Attribution graph
            metrics: Computed metrics
            task_features: Task-specific features found

        Returns:
            Interpretation string
        """
        interpretation_parts = []

        # Overall assessment
        weighted_score = metrics.get("weighted_score", 0)
        if weighted_score > 0.7:
            interpretation_parts.append("Strong alignment with task requirements.")
        elif weighted_score > 0.5:
            interpretation_parts.append("Moderate alignment with task requirements.")
        else:
            interpretation_parts.append("Weak alignment with task requirements.")

        # Feature activity
        n_features = task_features.get("n_active_features", 0)
        interpretation_parts.append(f"{n_features} features active.")

        # Coherence assessment
        coherence = metrics.get("coherence", 0)
        if coherence > 0.7:
            interpretation_parts.append("Pathways are highly coherent.")
        elif coherence > 0.5:
            interpretation_parts.append("Pathways show moderate coherence.")
        else:
            interpretation_parts.append("Pathways are diffuse.")

        return " ".join(interpretation_parts)

    def _compute_aggregate_metrics(
        self, results: list[EvaluationResult]
    ) -> dict[str, float]:
        """Compute aggregate metrics across all results.

        Args:
            results: List of individual evaluation results

        Returns:
            Dictionary of aggregate metric values
        """
        if not results:
            return {}

        # Get all metric names
        metric_names = set()
        for result in results:
            metric_names.update(result.metrics.keys())

        # Compute means
        aggregates = {}
        for metric_name in metric_names:
            values = [r.metrics.get(metric_name, 0) for r in results]
            aggregates[f"{metric_name}_mean"] = sum(values) / len(values)
            aggregates[f"{metric_name}_min"] = min(values)
            aggregates[f"{metric_name}_max"] = max(values)

        # Add count statistics
        aggregates["total_prompts"] = len(results)
        aggregates["avg_active_features"] = sum(
            r.task_features.get("n_active_features", 0) for r in results
        ) / len(results)

        return aggregates


from dataclasses import dataclass


@dataclass
class MetricConfig:
    """Default metric configuration."""

    weight: float = 1.0
    description: str = ""
    enabled: bool = True
