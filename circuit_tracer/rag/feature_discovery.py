"""Task Feature Discovery: Automatically find task-critical features."""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from circuit_tracer import ReplacementModel, attribute

logger = logging.getLogger(__name__)


class TaskFeatureDiscovery:
    """Automatically discover features critical for a specific task.

    Identifies task-relevant features by comparing:
    - Task prompts (what you want the model to do)
    - Baseline prompts (general behavior)

    Features that activate strongly on task but not baseline are task-critical.

    Use cases:
    - Automatic task configuration generation
    - Monitor fine-tuning (check if task features emerge)
    - Create reusable feature signatures
    - Understand what makes a task work
    """

    def __init__(
        self,
        model: ReplacementModel,
        attribution_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize the task feature discovery tool.

        Args:
            model: ReplacementModel to analyze
            attribution_kwargs: Additional kwargs for attribute()
        """
        self.model = model
        self.attribution_kwargs = attribution_kwargs or {}

    def discover_critical_features(
        self,
        task_prompts: list[str],
        baseline_prompts: list[str],
        top_k: int = 20,
        min_task_frequency: float = 0.3,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Find features that activate strongly on task but not baseline.

        Args:
            task_prompts: Prompts for the target task
            baseline_prompts: General prompts for comparison
            top_k: Number of top features to return
            min_task_frequency: Minimum frequency in task prompts (0-1)
            verbose: Whether to show progress

        Returns:
            Dictionary with discovered features and statistics
        """
        if len(task_prompts) < 3:
            logger.warning("Few task prompts - results may be noisy. Recommend 10+ prompts.")
        if len(baseline_prompts) < 3:
            logger.warning(
                "Few baseline prompts - results may be noisy. Recommend 10+ prompts."
            )

        # Run attribution on all prompts
        if verbose:
            logger.info("Analyzing task prompts...")
        task_graphs = [
            attribute(p, self.model, **self.attribution_kwargs)
            for p in tqdm(task_prompts, desc="Task prompts") if verbose else task_prompts
        ]

        if verbose:
            logger.info("Analyzing baseline prompts...")
        baseline_graphs = [
            attribute(p, self.model, **self.attribution_kwargs)
            for p in tqdm(baseline_prompts, desc="Baseline prompts")
            if verbose
            else baseline_prompts
        ]

        # Collect feature statistics
        task_feature_counts = defaultdict(int)
        task_feature_strength = defaultdict(float)
        baseline_feature_counts = defaultdict(int)

        for graph in task_graphs:
            for i, feat in enumerate(graph.active_features):
                feat_tuple = tuple(feat.tolist())
                task_feature_counts[feat_tuple] += 1
                task_feature_strength[feat_tuple] += graph.activation_values[i].item()

        for graph in baseline_graphs:
            for feat in graph.active_features:
                feat_tuple = tuple(feat.tolist())
                baseline_feature_counts[feat_tuple] += 1

        # Score features by task-specificity
        feature_scores = {}
        for feat, task_count in task_feature_counts.items():
            # Filter by minimum task frequency
            task_freq = task_count / len(task_prompts)
            if task_freq < min_task_frequency:
                continue

            baseline_count = baseline_feature_counts.get(feat, 0)
            baseline_freq = (
                baseline_count / len(baseline_prompts) if baseline_count > 0 else 0.01
            )

            # Task-specificity score
            # High if: frequent in task, rare in baseline, strong activation
            specificity = task_freq / baseline_freq
            avg_strength = task_feature_strength[feat] / task_count

            # Combined score
            feature_scores[feat] = specificity * avg_strength

        # Sort and return top-k
        sorted_features = sorted(
            feature_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k]

        return {
            "features": [f[0] for f in sorted_features],
            "scores": [f[1] for f in sorted_features],
            "task_frequency": {
                f: task_feature_counts[f] / len(task_prompts) for f, _ in sorted_features
            },
            "baseline_frequency": {
                f: baseline_feature_counts.get(f, 0) / len(baseline_prompts)
                for f, _ in sorted_features
            },
            "avg_activation_strength": {
                f: task_feature_strength[f] / task_feature_counts[f]
                for f, _ in sorted_features
            },
            "task_prompts_count": len(task_prompts),
            "baseline_prompts_count": len(baseline_prompts),
        }

    def create_feature_signature(
        self,
        task_name: str,
        critical_features: list[tuple],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a reusable feature signature for a task.

        Args:
            task_name: Name of the task
            critical_features: List of (layer, pos, feat_idx) tuples
            metadata: Additional metadata to store

        Returns:
            Feature signature dictionary
        """
        return {
            "task_name": task_name,
            "features": critical_features,
            "feature_count": len(critical_features),
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

    def save_signature(self, signature: dict[str, Any], output_path: str | Path):
        """Save feature signature to JSON file.

        Args:
            signature: Feature signature from create_feature_signature()
            output_path: Path to save JSON
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert tuples to lists for JSON serialization
        signature_copy = signature.copy()
        signature_copy["features"] = [list(f) for f in signature["features"]]

        with open(output_path, "w") as f:
            json.dump(signature_copy, f, indent=2)

        logger.info(f"Saved feature signature to {output_path}")

    def load_signature(self, signature_path: str | Path) -> dict[str, Any]:
        """Load feature signature from JSON file.

        Args:
            signature_path: Path to signature JSON

        Returns:
            Feature signature
        """
        with open(signature_path) as f:
            signature = json.load(f)

        # Convert lists back to tuples
        signature["features"] = [tuple(f) for f in signature["features"]]

        return signature

    def compare_signatures(
        self,
        signature1: dict[str, Any],
        signature2: dict[str, Any],
    ) -> dict[str, Any]:
        """Compare two feature signatures.

        Args:
            signature1: First signature
            signature2: Second signature

        Returns:
            Comparison metrics
        """
        features1 = set(signature1["features"])
        features2 = set(signature2["features"])

        intersection = features1 & features2
        union = features1 | features2

        jaccard = len(intersection) / len(union) if union else 0
        overlap_pct = len(intersection) / min(len(features1), len(features2))

        return {
            "task1": signature1["task_name"],
            "task2": signature2["task_name"],
            "shared_features": len(intersection),
            "unique_to_task1": len(features1 - features2),
            "unique_to_task2": len(features2 - features1),
            "jaccard_similarity": jaccard,
            "overlap_percentage": overlap_pct,
        }

    def track_feature_evolution(
        self,
        task_prompts: list[str],
        baseline_prompts: list[str],
        checkpoints: list[tuple[str, ReplacementModel]],
        top_k: int = 15,
    ) -> dict[str, Any]:
        """Track how task features evolve across model checkpoints (e.g., during fine-tuning).

        Args:
            task_prompts: Task prompts to evaluate
            baseline_prompts: Baseline prompts
            checkpoints: List of (name, model) tuples
            top_k: Number of top features to track

        Returns:
            Feature evolution analysis
        """
        evolution = {
            "checkpoints": [],
            "feature_stability": {},
        }

        all_features_seen = set()

        for ckpt_name, ckpt_model in checkpoints:
            # Temporarily swap model
            original_model = self.model
            self.model = ckpt_model

            # Discover features
            result = self.discover_critical_features(
                task_prompts=task_prompts,
                baseline_prompts=baseline_prompts,
                top_k=top_k,
                verbose=False,
            )

            evolution["checkpoints"].append({
                "name": ckpt_name,
                "features": result["features"],
                "scores": result["scores"],
            })

            all_features_seen.update(result["features"])

            # Restore model
            self.model = original_model

        # Analyze feature stability
        for feat in all_features_seen:
            appearances = sum(
                1 for ckpt in evolution["checkpoints"] if feat in ckpt["features"]
            )
            evolution["feature_stability"][feat] = appearances / len(checkpoints)

        # Identify stable vs emerging vs disappearing features
        stable = [
            f for f, stability in evolution["feature_stability"].items() if stability > 0.8
        ]
        emerging = [
            f
            for f in evolution["checkpoints"][-1]["features"]
            if f not in evolution["checkpoints"][0]["features"]
        ]
        disappearing = [
            f
            for f in evolution["checkpoints"][0]["features"]
            if f not in evolution["checkpoints"][-1]["features"]
        ]

        evolution["analysis"] = {
            "stable_features": stable,
            "emerging_features": emerging,
            "disappearing_features": disappearing,
            "stability_mean": sum(evolution["feature_stability"].values())
            / len(evolution["feature_stability"])
            if evolution["feature_stability"]
            else 0,
        }

        return evolution

    def visualize_features(
        self, discovery_result: dict[str, Any], output_path: str | Path | None = None
    ) -> str:
        """Generate a text visualization of discovered features.

        Args:
            discovery_result: Result from discover_critical_features()
            output_path: Optional path to save visualization

        Returns:
            Visualization string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("DISCOVERED TASK-CRITICAL FEATURES")
        lines.append("=" * 70)

        for i, (feat, score) in enumerate(
            zip(discovery_result["features"], discovery_result["scores"])
        ):
            layer, pos, feat_idx = feat
            task_freq = discovery_result["task_frequency"][feat]
            baseline_freq = discovery_result["baseline_frequency"][feat]
            strength = discovery_result["avg_activation_strength"][feat]

            lines.append(f"\n{i+1}. Feature (L{layer}, P{pos}, F{feat_idx})")
            lines.append(f"   Score: {score:.3f}")
            lines.append(f"   Task frequency: {task_freq:.1%}")
            lines.append(f"   Baseline frequency: {baseline_freq:.1%}")
            lines.append(f"   Avg activation: {strength:.3f}")

        lines.append("\n" + "=" * 70)
        lines.append(
            f"Total: {len(discovery_result['features'])} features discovered"
        )
        lines.append(
            f"From {discovery_result['task_prompts_count']} task prompts, "
            f"{discovery_result['baseline_prompts_count']} baseline prompts"
        )
        lines.append("=" * 70)

        viz = "\n".join(lines)

        if output_path:
            with open(output_path, "w") as f:
                f.write(viz)
            logger.info(f"Saved visualization to {output_path}")

        return viz
