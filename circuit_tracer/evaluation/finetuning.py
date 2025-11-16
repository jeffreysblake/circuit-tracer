"""Fine-tuning evaluation: Compare circuits between base and fine-tuned models."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.evaluation.metrics import (
    compute_feature_coverage,
    compute_pathway_coherence,
    compute_sparsity,
)
from circuit_tracer.evaluation.task_config import TaskConfig
from circuit_tracer.evaluation.task_evaluator import EvaluationResult, TaskEvaluator
from circuit_tracer.graph import Graph
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

logger = logging.getLogger(__name__)


@dataclass
class FeatureDrift:
    """Metrics quantifying how much features have changed."""

    feature_overlap: float  # % of base features still active
    activation_correlation: float  # Correlation of activation patterns
    transcoder_validity: float  # Reconstruction quality
    new_features_count: int  # Features unique to fine-tuned model
    lost_features_count: int  # Base features no longer active
    strengthened_features: list[tuple[int, int, int]]  # (layer, pos, feat_idx)
    weakened_features: list[tuple[int, int, int]]


@dataclass
class CheckpointComparison:
    """Comparison between base model and a checkpoint."""

    checkpoint_name: str
    checkpoint_step: int | None
    base_results: dict[str, Any]
    checkpoint_results: dict[str, Any]
    drift_metrics: FeatureDrift
    metric_deltas: dict[str, float]  # Change in each metric
    interpretation: str = ""


@dataclass
class FineTuningAnalysis:
    """Complete analysis of fine-tuning progression."""

    base_model_name: str
    transcoder_set: str
    task_name: str
    checkpoints: list[CheckpointComparison] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


class FineTuneComparator:
    """Compare circuits between base model and fine-tuned checkpoints."""

    def __init__(
        self,
        base_model: str,
        transcoder_set: str,
        task_config: TaskConfig,
        dtype: torch.dtype = torch.float32,
        attribution_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize the fine-tune comparator.

        Args:
            base_model: HuggingFace model name for base model
            transcoder_set: Transcoder set to use (trained on base model)
            task_config: Task configuration for evaluation
            dtype: Data type for model loading
            attribution_kwargs: Additional kwargs for attribution
        """
        self.base_model_name = base_model
        self.transcoder_set = transcoder_set
        self.task_config = task_config
        self.dtype = dtype
        self.attribution_kwargs = attribution_kwargs or {}

        # Load base model and transcoders once
        logger.info(f"Loading base model: {base_model}")
        transcoder, config = load_transcoder_from_hub(transcoder_set, dtype=dtype)
        self.base_model = ReplacementModel.from_pretrained_and_transcoders(
            base_model, transcoder, dtype=dtype
        )

        self.base_evaluator = TaskEvaluator(
            model=self.base_model,
            task_config=task_config,
            attribution_kwargs=self.attribution_kwargs,
        )

        # Cache for base model evaluation (so we don't re-run it each time)
        self.base_cache: dict[str, EvaluationResult] = {}

    def evaluate_checkpoint(
        self,
        checkpoint_path: str,
        eval_prompts: list[str],
        checkpoint_name: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate a single checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
            eval_prompts: Prompts to evaluate on
            checkpoint_name: Optional name for the checkpoint

        Returns:
            Evaluation results dictionary
        """
        checkpoint_name = checkpoint_name or Path(checkpoint_path).name

        logger.info(f"Evaluating checkpoint: {checkpoint_name}")

        # Load checkpoint with same transcoders as base
        transcoder, _ = load_transcoder_from_hub(self.transcoder_set, dtype=self.dtype)
        checkpoint_model = ReplacementModel.from_pretrained_and_transcoders(
            checkpoint_path, transcoder, dtype=self.dtype
        )

        # Create evaluator for checkpoint
        checkpoint_evaluator = TaskEvaluator(
            model=checkpoint_model,
            task_config=self.task_config,
            attribution_kwargs=self.attribution_kwargs,
        )

        # Run evaluation
        results = checkpoint_evaluator.evaluate_batch(
            prompts=eval_prompts, verbose=False, save_graphs=True
        )

        return {
            "checkpoint_name": checkpoint_name,
            "metrics": results.aggregate_metrics,
            "results": results,
        }

    def compute_drift(
        self,
        base_graph: Graph,
        checkpoint_graph: Graph,
    ) -> FeatureDrift:
        """Compute feature drift between base and checkpoint.

        Args:
            base_graph: Attribution graph from base model
            checkpoint_graph: Attribution graph from checkpoint

        Returns:
            FeatureDrift metrics
        """
        base_features = set(
            tuple(f.tolist()) for f in base_graph.active_features
        )
        ckpt_features = set(
            tuple(f.tolist()) for f in checkpoint_graph.active_features
        )

        # Feature overlap
        overlap = len(base_features & ckpt_features)
        feature_overlap = overlap / len(base_features) if base_features else 0.0

        # New and lost features
        new_features = ckpt_features - base_features
        lost_features = base_features - ckpt_features

        # Activation correlation (for overlapping features)
        overlapping_features = base_features & ckpt_features
        if overlapping_features:
            # Build mapping from feature tuple to activation value
            base_activations = {
                tuple(base_graph.active_features[i].tolist()): base_graph.activation_values[i].item()
                for i in range(len(base_graph.active_features))
            }
            ckpt_activations = {
                tuple(checkpoint_graph.active_features[i].tolist()): checkpoint_graph.activation_values[
                    i
                ].item()
                for i in range(len(checkpoint_graph.active_features))
            }

            # Compute correlation for overlapping features
            base_vals = [base_activations[f] for f in overlapping_features]
            ckpt_vals = [ckpt_activations[f] for f in overlapping_features]

            base_tensor = torch.tensor(base_vals)
            ckpt_tensor = torch.tensor(ckpt_vals)
            correlation = torch.corrcoef(torch.stack([base_tensor, ckpt_tensor]))[0, 1].item()
        else:
            correlation = 0.0

        # Transcoder validity: compare reconstruction quality
        # Higher validity = transcoders still work well
        # This is a simplified metric - ideally would measure actual reconstruction error
        transcoder_validity = feature_overlap * correlation

        # Find strengthened/weakened features
        strengthened = []
        weakened = []

        for feat in overlapping_features:
            base_val = base_activations[feat]
            ckpt_val = ckpt_activations[feat]
            change_ratio = ckpt_val / (base_val + 1e-8)

            if change_ratio > 1.5:  # 50% stronger
                strengthened.append(feat)
            elif change_ratio < 0.67:  # 33% weaker
                weakened.append(feat)

        return FeatureDrift(
            feature_overlap=feature_overlap,
            activation_correlation=correlation,
            transcoder_validity=transcoder_validity,
            new_features_count=len(new_features),
            lost_features_count=len(lost_features),
            strengthened_features=strengthened[:20],  # Top 20
            weakened_features=weakened[:20],
        )

    def compare_checkpoints(
        self,
        checkpoint_paths: list[str],
        eval_prompts: list[str],
        baseline_prompts: list[str] | None = None,
    ) -> FineTuningAnalysis:
        """Compare multiple checkpoints against base model.

        Args:
            checkpoint_paths: List of checkpoint directories
            eval_prompts: Task-specific prompts to evaluate
            baseline_prompts: Optional baseline prompts for comparison

        Returns:
            FineTuningAnalysis with all comparisons
        """
        analysis = FineTuningAnalysis(
            base_model_name=self.base_model_name,
            transcoder_set=self.transcoder_set,
            task_name=self.task_config.task_name,
        )

        # Evaluate base model once
        logger.info("Evaluating base model...")
        base_results = self.base_evaluator.evaluate_batch(
            prompts=eval_prompts, verbose=False, save_graphs=True
        )

        # Compare each checkpoint
        for ckpt_path in tqdm(checkpoint_paths, desc="Evaluating checkpoints"):
            ckpt_name = Path(ckpt_path).name
            ckpt_step = self._extract_step_number(ckpt_name)

            # Evaluate checkpoint
            ckpt_eval = self.evaluate_checkpoint(ckpt_path, eval_prompts, ckpt_name)

            # Compute drift metrics (using first prompt's graphs as representative)
            base_graph = base_results.results[0].graph
            ckpt_graph = ckpt_eval["results"].results[0].graph

            drift = self.compute_drift(base_graph, ckpt_graph)

            # Compute metric deltas
            metric_deltas = {}
            for metric_name in ["coverage", "coherence", "sparsity", "weighted_score"]:
                base_val = base_results.aggregate_metrics.get(f"{metric_name}_mean", 0)
                ckpt_val = ckpt_eval["metrics"].get(f"{metric_name}_mean", 0)
                metric_deltas[metric_name] = ckpt_val - base_val

            # Generate interpretation
            interpretation = self._interpret_checkpoint(drift, metric_deltas)

            comparison = CheckpointComparison(
                checkpoint_name=ckpt_name,
                checkpoint_step=ckpt_step,
                base_results=base_results.summary(),
                checkpoint_results=ckpt_eval["results"].summary(),
                drift_metrics=drift,
                metric_deltas=metric_deltas,
                interpretation=interpretation,
            )

            analysis.checkpoints.append(comparison)

        # Generate summary
        analysis.summary = self._generate_summary(analysis)

        return analysis

    def _extract_step_number(self, checkpoint_name: str) -> int | None:
        """Extract step number from checkpoint name.

        Args:
            checkpoint_name: Name like "checkpoint-1000"

        Returns:
            Step number or None
        """
        import re

        match = re.search(r"(\d+)", checkpoint_name)
        return int(match.group(1)) if match else None

    def _interpret_checkpoint(
        self, drift: FeatureDrift, metric_deltas: dict[str, float]
    ) -> str:
        """Generate human-readable interpretation of checkpoint.

        Args:
            drift: Drift metrics
            metric_deltas: Changes in evaluation metrics

        Returns:
            Interpretation string
        """
        parts = []

        # Transcoder validity
        if drift.transcoder_validity > 0.8:
            parts.append("✅ Transcoders remain highly valid.")
        elif drift.transcoder_validity > 0.6:
            parts.append("⚠️ Transcoders moderately degraded.")
        else:
            parts.append("❌ Transcoders significantly degraded - consider retraining.")

        # Feature changes
        if drift.new_features_count > drift.lost_features_count * 2:
            parts.append(f"Learned {drift.new_features_count} new features.")
        if drift.lost_features_count > 10:
            parts.append(
                f"⚠️ Lost {drift.lost_features_count} base features (possible forgetting)."
            )

        # Metric improvements
        improved = [k for k, v in metric_deltas.items() if v > 0.05]
        degraded = [k for k, v in metric_deltas.items() if v < -0.05]

        if improved:
            parts.append(f"Improved: {', '.join(improved)}.")
        if degraded:
            parts.append(f"⚠️ Degraded: {', '.join(degraded)}.")

        return " ".join(parts)

    def _generate_summary(self, analysis: FineTuningAnalysis) -> dict[str, Any]:
        """Generate summary statistics across all checkpoints.

        Args:
            analysis: Analysis with checkpoint comparisons

        Returns:
            Summary dictionary
        """
        if not analysis.checkpoints:
            return {}

        # Track best checkpoint for each metric
        best_coverage_idx = max(
            range(len(analysis.checkpoints)),
            key=lambda i: analysis.checkpoints[i].metric_deltas.get("coverage", -999),
        )
        best_coherence_idx = max(
            range(len(analysis.checkpoints)),
            key=lambda i: analysis.checkpoints[i].metric_deltas.get("coherence", -999),
        )

        # Track transcoder validity over time
        validity_trend = [
            ckpt.drift_metrics.transcoder_validity for ckpt in analysis.checkpoints
        ]

        return {
            "total_checkpoints": len(analysis.checkpoints),
            "best_coverage_checkpoint": analysis.checkpoints[best_coverage_idx].checkpoint_name,
            "best_coherence_checkpoint": analysis.checkpoints[best_coherence_idx].checkpoint_name,
            "final_validity": validity_trend[-1] if validity_trend else 0,
            "validity_trend": "improving" if len(validity_trend) > 1 and validity_trend[-1] > validity_trend[0] else "degrading",
            "avg_new_features": sum(
                ckpt.drift_metrics.new_features_count for ckpt in analysis.checkpoints
            )
            / len(analysis.checkpoints),
        }

    def save_analysis(self, analysis: FineTuningAnalysis, output_path: str):
        """Save analysis to JSON file.

        Args:
            analysis: Analysis to save
            output_path: Path to save JSON
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-serializable format
        data = {
            "base_model": analysis.base_model_name,
            "transcoder_set": analysis.transcoder_set,
            "task_name": analysis.task_name,
            "summary": analysis.summary,
            "checkpoints": [
                {
                    "name": ckpt.checkpoint_name,
                    "step": ckpt.checkpoint_step,
                    "drift": {
                        "feature_overlap": ckpt.drift_metrics.feature_overlap,
                        "activation_correlation": ckpt.drift_metrics.activation_correlation,
                        "transcoder_validity": ckpt.drift_metrics.transcoder_validity,
                        "new_features": ckpt.drift_metrics.new_features_count,
                        "lost_features": ckpt.drift_metrics.lost_features_count,
                    },
                    "metric_deltas": ckpt.metric_deltas,
                    "interpretation": ckpt.interpretation,
                }
                for ckpt in analysis.checkpoints
            ],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Analysis saved to {output_path}")
