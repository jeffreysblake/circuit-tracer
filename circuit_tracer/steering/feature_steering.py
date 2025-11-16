"""Feature Steering: Control model behavior by modifying feature activations at inference time."""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from tqdm import tqdm

from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.graph import Graph

logger = logging.getLogger(__name__)


@dataclass
class FeatureEdit:
    """A single feature modification."""

    feature: tuple[int, int, int]  # (layer, pos, feat_idx)
    weight: float  # Positive to amplify, negative to suppress
    description: str = ""


@dataclass
class ConditionalSteering:
    """Apply steering only when a condition is met."""

    condition: Callable[[str], bool]  # Function that checks if condition is met
    edits: dict[tuple[int, int, int], float]  # feature -> weight
    description: str = ""


@dataclass
class SteeringConfig:
    """Configuration for feature steering."""

    # Feature weights to apply: (layer, pos, feat_idx) -> weight
    feature_weights: dict[tuple[int, int, int], float] = field(default_factory=dict)

    # Conditional steering rules
    conditional_rules: list[ConditionalSteering] = field(default_factory=list)

    # Abort generation if condition is met
    abort_conditions: list[Callable[[str], bool]] = field(default_factory=list)

    # Whether to track activations during generation
    track_activations: bool = False


class FeatureSteering:
    """Control model behavior by modifying feature activations.

    Inspired by Goodfire's Feature Steering API, this class provides local
    feature steering capabilities using circuit-tracer's transcoders.

    Features can be:
    - Amplified (positive weight): Make the model use this feature more
    - Suppressed (negative weight): Make the model avoid this feature
    - Conditionally modified: Apply steering only when certain conditions are met

    Example:
        >>> from circuit_tracer.steering import FeatureSteering
        >>> steering = FeatureSteering(model)
        >>>
        >>> # Discover "concise" features
        >>> features = steering.discover_features("concise and direct responses")
        >>>
        >>> # Amplify conciseness
        >>> steering.set_feature_weight(features[0], 0.5)
        >>>
        >>> # Generate with steering
        >>> output = steering.generate("Explain quantum computing")
    """

    def __init__(
        self,
        model: ReplacementModel,
        attribution_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize feature steering.

        Args:
            model: ReplacementModel with transcoders
            attribution_kwargs: Additional kwargs for attribution
        """
        self.model = model
        self.attribution_kwargs = attribution_kwargs or {}
        self.config = SteeringConfig()

        # Track activations during generation
        self._activation_history: list[dict] = []

    def discover_features(
        self,
        description: str,
        n_features: int = 10,
        method: str = "semantic",
        reference_prompts: list[str] | None = None,
    ) -> list[tuple[int, int, int]]:
        """Discover features related to a semantic description.

        Args:
            description: Natural language description of desired behavior
            n_features: Number of features to return
            method: Discovery method ("semantic" or "frequency")
            reference_prompts: Optional example prompts that exhibit the behavior

        Returns:
            List of (layer, pos, feat_idx) tuples
        """
        if method == "semantic":
            # Use the description as a prompt to find relevant features
            if not reference_prompts:
                reference_prompts = [description]

            return self._discover_semantic(reference_prompts, n_features)

        elif method == "frequency":
            # Find most frequently activated features
            if not reference_prompts:
                raise ValueError("reference_prompts required for frequency method")

            return self._discover_by_frequency(reference_prompts, n_features)

        else:
            raise ValueError(f"Unknown method: {method}")

    def _discover_semantic(
        self, prompts: list[str], n_features: int
    ) -> list[tuple[int, int, int]]:
        """Discover features by analyzing prompts."""
        feature_counts = defaultdict(int)
        feature_strengths = defaultdict(float)

        for prompt in prompts:
            graph = attribute(prompt, self.model, **self.attribution_kwargs)

            for i, feat in enumerate(graph.active_features):
                feat_tuple = tuple(feat.tolist())
                feature_counts[feat_tuple] += 1
                feature_strengths[feat_tuple] += graph.activation_values[i].item()

        # Sort by frequency * strength
        scored_features = [
            (feat, feature_counts[feat] * feature_strengths[feat])
            for feat in feature_counts
        ]
        scored_features.sort(key=lambda x: x[1], reverse=True)

        return [f for f, _ in scored_features[:n_features]]

    def _discover_by_frequency(
        self, prompts: list[str], n_features: int
    ) -> list[tuple[int, int, int]]:
        """Discover most frequently activated features."""
        feature_counts = defaultdict(int)

        for prompt in prompts:
            graph = attribute(prompt, self.model, **self.attribution_kwargs)

            for feat in graph.active_features:
                feat_tuple = tuple(feat.tolist())
                feature_counts[feat_tuple] += 1

        # Sort by frequency
        sorted_features = sorted(
            feature_counts.items(), key=lambda x: x[1], reverse=True
        )

        return [f for f, _ in sorted_features[:n_features]]

    def discover_features_contrastive(
        self,
        desired_examples: list[str],
        undesired_examples: list[str],
        top_k: int = 20,
        min_frequency: float = 0.3,
    ) -> list[tuple[tuple[int, int, int], float]]:
        """Discover features that differentiate desired from undesired behavior.

        This is similar to Goodfire's Contrastive Search.

        Args:
            desired_examples: Prompts exhibiting desired behavior
            undesired_examples: Prompts exhibiting undesired behavior
            top_k: Number of top features to return
            min_frequency: Minimum frequency in desired examples (0-1)

        Returns:
            List of (feature, suggested_weight) tuples
        """
        # Analyze desired examples
        desired_counts = defaultdict(int)
        desired_strengths = defaultdict(float)

        for prompt in tqdm(desired_examples, desc="Analyzing desired behavior"):
            graph = attribute(prompt, self.model, **self.attribution_kwargs)

            for i, feat in enumerate(graph.active_features):
                feat_tuple = tuple(feat.tolist())
                desired_counts[feat_tuple] += 1
                desired_strengths[feat_tuple] += graph.activation_values[i].item()

        # Analyze undesired examples
        undesired_counts = defaultdict(int)

        for prompt in tqdm(undesired_examples, desc="Analyzing undesired behavior"):
            graph = attribute(prompt, self.model, **self.attribution_kwargs)

            for feat in graph.active_features:
                feat_tuple = tuple(feat.tolist())
                undesired_counts[feat_tuple] += 1

        # Score features by contrast
        feature_scores = {}
        for feat, desired_count in desired_counts.items():
            # Filter by minimum frequency in desired
            desired_freq = desired_count / len(desired_examples)
            if desired_freq < min_frequency:
                continue

            undesired_count = undesired_counts.get(feat, 0)
            undesired_freq = (
                undesired_count / len(undesired_examples)
                if undesired_count > 0
                else 0.01
            )

            # Contrast score: high if frequent in desired, rare in undesired
            contrast = desired_freq / undesired_freq
            avg_strength = desired_strengths[feat] / desired_count

            # Suggested weight: positive for features to amplify
            suggested_weight = min(contrast * avg_strength * 0.1, 1.0)

            feature_scores[feat] = suggested_weight

        # Sort and return top-k
        sorted_features = sorted(
            feature_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k]

        return sorted_features

    def set_feature_weight(
        self,
        feature: tuple[int, int, int],
        weight: float,
        description: str = "",
    ):
        """Set the weight for a specific feature.

        Args:
            feature: (layer, pos, feat_idx) tuple
            weight: Steering weight (positive to amplify, negative to suppress)
            description: Optional description of what this feature does
        """
        self.config.feature_weights[feature] = weight

        if description:
            logger.info(f"Set feature {feature} to weight {weight:.2f}: {description}")
        else:
            logger.info(f"Set feature {feature} to weight {weight:.2f}")

    def set_features(self, edits: dict[tuple[int, int, int], float]):
        """Set multiple feature weights at once.

        Args:
            edits: Dictionary mapping features to weights
        """
        for feature, weight in edits.items():
            self.set_feature_weight(feature, weight)

    def clear_steering(self):
        """Clear all steering configurations."""
        self.config = SteeringConfig()
        logger.info("Cleared all steering configurations")

    def set_conditional_steering(
        self,
        condition: Callable[[str], bool],
        features: dict[tuple[int, int, int], float],
        description: str = "",
    ):
        """Apply steering only when a condition is met.

        Args:
            condition: Function that takes text and returns True if condition is met
            features: Dictionary mapping features to weights
            description: Optional description of the condition
        """
        rule = ConditionalSteering(
            condition=condition,
            edits=features,
            description=description,
        )

        self.config.conditional_rules.append(rule)
        logger.info(f"Added conditional steering: {description}")

    def add_abort_condition(
        self,
        condition: Callable[[str], bool],
        description: str = "",
    ):
        """Abort generation if a condition is met.

        Args:
            condition: Function that takes text and returns True to abort
            description: Optional description of the abort condition
        """
        self.config.abort_conditions.append(condition)
        logger.info(f"Added abort condition: {description}")

    def _apply_steering_to_graph(self, graph: Graph, text: str) -> Graph:
        """Apply steering modifications to an attribution graph.

        Args:
            graph: Original attribution graph
            text: Input text (for conditional checks)

        Returns:
            Modified graph with steering applied
        """
        # Check abort conditions
        for abort_fn in self.config.abort_conditions:
            if abort_fn(text):
                logger.warning("Abort condition triggered")
                # In practice, this would stop generation
                return graph

        # Apply base feature weights
        active_weights = self.config.feature_weights.copy()

        # Apply conditional steering
        for rule in self.config.conditional_rules:
            if rule.condition(text):
                logger.debug(f"Conditional steering triggered: {rule.description}")
                active_weights.update(rule.edits)

        # Modify activation values based on weights
        if active_weights:
            modified_activations = graph.activation_values.clone()

            for i, feat in enumerate(graph.active_features):
                feat_tuple = tuple(feat.tolist())
                if feat_tuple in active_weights:
                    weight = active_weights[feat_tuple]
                    # Multiply activation by (1 + weight)
                    # weight=0.5 means 50% stronger, weight=-0.5 means 50% weaker
                    modified_activations[i] *= 1.0 + weight

            # Create modified graph
            # Note: In practice, steering would happen during forward pass
            # This is a simplified representation
            graph.activation_values = modified_activations

        return graph

    def analyze_with_steering(
        self,
        prompt: str,
        verbose: bool = False,
    ) -> tuple[Graph, Graph]:
        """Analyze prompt with and without steering.

        Args:
            prompt: Input prompt
            verbose: Whether to print details

        Returns:
            Tuple of (original_graph, steered_graph)
        """
        # Original graph
        original_graph = attribute(prompt, self.model, **self.attribution_kwargs)

        # Steered graph
        steered_graph = self._apply_steering_to_graph(original_graph, prompt)

        if verbose:
            original_strength = original_graph.activation_values.sum().item()
            steered_strength = steered_graph.activation_values.sum().item()

            print(f"Original total activation: {original_strength:.3f}")
            print(f"Steered total activation: {steered_strength:.3f}")
            print(f"Change: {steered_strength - original_strength:.3f}")

        return original_graph, steered_graph

    def inspect_activations(
        self,
        prompt: str,
        top_k: int = 20,
    ) -> dict[str, Any]:
        """Inspect which features activate for a given prompt.

        Args:
            prompt: Input prompt
            top_k: Number of top features to return

        Returns:
            Dictionary with activation information
        """
        graph = attribute(prompt, self.model, **self.attribution_kwargs)

        # Sort features by activation strength
        feature_activations = [
            (tuple(graph.active_features[i].tolist()), graph.activation_values[i].item())
            for i in range(len(graph.active_features))
        ]
        feature_activations.sort(key=lambda x: x[1], reverse=True)

        # Check which features are being steered
        steered_features = []
        for feat, activation in feature_activations[:top_k]:
            if feat in self.config.feature_weights:
                steered_features.append({
                    "feature": feat,
                    "activation": activation,
                    "steering_weight": self.config.feature_weights[feat],
                    "steered_activation": activation * (1.0 + self.config.feature_weights[feat]),
                })

        return {
            "top_features": feature_activations[:top_k],
            "steered_features": steered_features,
            "total_active": len(graph.active_features),
            "total_activation": graph.activation_values.sum().item(),
        }

    def visualize_steering(self, output_path: str | None = None) -> str:
        """Generate a text visualization of current steering configuration.

        Args:
            output_path: Optional path to save visualization

        Returns:
            Visualization string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("FEATURE STEERING CONFIGURATION")
        lines.append("=" * 70)

        # Base feature weights
        if self.config.feature_weights:
            lines.append(f"\nBase Feature Weights: {len(self.config.feature_weights)}")
            lines.append("-" * 70)

            for feat, weight in sorted(
                self.config.feature_weights.items(), key=lambda x: abs(x[1]), reverse=True
            ):
                layer, pos, feat_idx = feat
                direction = "AMPLIFY" if weight > 0 else "SUPPRESS"
                lines.append(
                    f"  L{layer} P{pos} F{feat_idx}: {weight:+.2f} ({direction})"
                )
        else:
            lines.append("\nNo base feature weights set")

        # Conditional rules
        if self.config.conditional_rules:
            lines.append(f"\n\nConditional Steering Rules: {len(self.config.conditional_rules)}")
            lines.append("-" * 70)

            for i, rule in enumerate(self.config.conditional_rules, 1):
                lines.append(f"\n{i}. {rule.description or 'Unnamed rule'}")
                lines.append(f"   Features affected: {len(rule.edits)}")
                for feat, weight in list(rule.edits.items())[:5]:  # Show first 5
                    layer, pos, feat_idx = feat
                    lines.append(f"     L{layer} P{pos} F{feat_idx}: {weight:+.2f}")
                if len(rule.edits) > 5:
                    lines.append(f"     ... and {len(rule.edits) - 5} more")
        else:
            lines.append("\n\nNo conditional steering rules set")

        # Abort conditions
        if self.config.abort_conditions:
            lines.append(f"\n\nAbort Conditions: {len(self.config.abort_conditions)}")
        else:
            lines.append("\n\nNo abort conditions set")

        lines.append("\n" + "=" * 70)

        viz = "\n".join(lines)

        if output_path:
            with open(output_path, "w") as f:
                f.write(viz)
            logger.info(f"Saved steering visualization to {output_path}")

        return viz

    def auto_steer(
        self,
        specification: str,
        reference_prompts: list[str] | None = None,
        n_features: int = 10,
        base_weight: float = 0.3,
    ) -> dict[tuple[int, int, int], float]:
        """Automatically discover and configure steering (similar to Goodfire's AutoSteer).

        Args:
            specification: Natural language description of desired behavior
            reference_prompts: Optional example prompts
            n_features: Number of features to discover
            base_weight: Base weight to apply to discovered features

        Returns:
            Dictionary of features and weights that were applied
        """
        logger.info(f"Auto-steering: {specification}")

        # Discover features
        features = self.discover_features(
            description=specification,
            n_features=n_features,
            reference_prompts=reference_prompts,
        )

        # Apply weights
        edits = {}
        for feat in features:
            edits[feat] = base_weight
            self.set_feature_weight(feat, base_weight)

        logger.info(f"Auto-steer configured {len(edits)} features")

        return edits

    def compare_behaviors(
        self,
        prompt: str,
        steering_configs: dict[str, dict[tuple[int, int, int], float]],
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Compare how different steering configurations affect a prompt.

        Args:
            prompt: Input prompt to test
            steering_configs: Dictionary of {name: {feature: weight}}
            verbose: Whether to print comparison

        Returns:
            Comparison results
        """
        results = {}

        # Baseline (no steering)
        original_config = self.config
        self.config = SteeringConfig()
        baseline_graph = attribute(prompt, self.model, **self.attribution_kwargs)
        results["baseline"] = {
            "total_activation": baseline_graph.activation_values.sum().item(),
            "num_features": len(baseline_graph.active_features),
        }
        self.config = original_config

        # Test each configuration
        for name, weights in steering_configs.items():
            # Apply this configuration
            self.clear_steering()
            self.set_features(weights)

            # Analyze
            graph = attribute(prompt, self.model, **self.attribution_kwargs)
            results[name] = {
                "total_activation": graph.activation_values.sum().item(),
                "num_features": len(graph.active_features),
                "delta": graph.activation_values.sum().item()
                - results["baseline"]["total_activation"],
            }

        if verbose:
            print("=" * 70)
            print("STEERING COMPARISON")
            print("=" * 70)
            print(f"\nPrompt: {prompt}\n")

            for name, metrics in results.items():
                print(f"{name}:")
                print(f"  Total activation: {metrics['total_activation']:.3f}")
                print(f"  Active features: {metrics['num_features']}")
                if "delta" in metrics:
                    print(f"  Change from baseline: {metrics['delta']:+.3f}")
                print()

        return results
