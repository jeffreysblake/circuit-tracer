"""Circuit-Guided Phrase Optimizer: Re-work phrases for better feature activation."""

import json
import logging
from collections.abc import Callable
from typing import Any, Literal

import torch
from tqdm import tqdm

from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.evaluation.metrics import compute_pathway_coherence
from circuit_tracer.graph import Graph

logger = logging.getLogger(__name__)


class CircuitGuidedPhraseOptimizer:
    """Optimize phrases to maximize task-relevant feature activation.

    This tool generates paraphrases and selects the best one based on
    circuit analysis rather than trial-and-error or task performance.

    Use cases:
    - Optimize RAG context formatting
    - Improve agent prompts
    - A/B test prompts with objective metrics
    - Find phrasings that activate specific features
    """

    def __init__(
        self,
        model: ReplacementModel,
        paraphrase_generator: Callable[[str, int], list[str]],
        attribution_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize the phrase optimizer.

        Args:
            model: ReplacementModel for circuit analysis
            paraphrase_generator: Function that takes (text, n) and returns n paraphrases
            attribution_kwargs: Additional kwargs for attribute()
        """
        self.model = model
        self.paraphrase_generator = paraphrase_generator
        self.attribution_kwargs = attribution_kwargs or {}

    def optimize_phrase(
        self,
        original_phrase: str,
        target_features: list[tuple] | None = None,
        n_variants: int = 5,
        optimization_metric: Literal["activation_strength", "feature_coverage", "balanced", "coherence"] = "balanced",
        include_original: bool = True,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Generate variants and select best by circuit activation.

        Args:
            original_phrase: Text to optimize
            target_features: List of (layer, pos, feat_idx) to target, or None for general optimization
            n_variants: Number of paraphrase variants to generate
            optimization_metric: How to score variants
            include_original: Whether to include original in comparison
            verbose: Whether to show progress

        Returns:
            Dictionary with optimized phrase and analysis
        """
        # Generate paraphrases
        logger.info(f"Generating {n_variants} paraphrases...")
        variants = self.paraphrase_generator(original_phrase, n_variants)

        if include_original:
            variants.append(original_phrase)

        # Evaluate each variant
        results = []
        iterator = tqdm(variants, desc="Evaluating variants") if verbose else variants

        for variant in iterator:
            try:
                graph = attribute(variant, self.model, **self.attribution_kwargs)

                score = self._score_variant(
                    graph=graph,
                    target_features=target_features,
                    metric=optimization_metric,
                )

                results.append({
                    "phrase": variant,
                    "score": score,
                    "active_features": len(graph.active_features),
                    "coherence": compute_pathway_coherence(graph),
                    "activation_strength": self._total_activation(graph, target_features) if target_features else 0,
                    "graph": graph,
                })
            except Exception as e:
                logger.warning(f"Failed to evaluate variant: {e}")
                continue

        if not results:
            raise ValueError("No variants could be evaluated")

        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)

        best = results[0]
        original_result = next((r for r in results if r["phrase"] == original_phrase), None)

        return {
            "optimized": best["phrase"],
            "original": original_phrase,
            "improvement": best["score"] / original_result["score"] if original_result else None,
            "best_score": best["score"],
            "original_score": original_result["score"] if original_result else None,
            "all_variants": results,
            "metric_used": optimization_metric,
        }

    def optimize_batch(
        self,
        phrases: list[str],
        target_features: list[tuple] | None = None,
        n_variants: int = 5,
        verbose: bool = True,
    ) -> list[dict[str, Any]]:
        """Optimize multiple phrases.

        Args:
            phrases: List of phrases to optimize
            target_features: Target features for all phrases
            n_variants: Number of variants per phrase
            verbose: Whether to show progress

        Returns:
            List of optimization results
        """
        results = []
        iterator = tqdm(phrases, desc="Optimizing phrases") if verbose else phrases

        for phrase in iterator:
            result = self.optimize_phrase(
                phrase,
                target_features=target_features,
                n_variants=n_variants,
                verbose=False,
            )
            results.append(result)

        return results

    def _score_variant(
        self,
        graph: Graph,
        target_features: list[tuple] | None,
        metric: str,
    ) -> float:
        """Score a variant based on circuit properties.

        Args:
            graph: Attribution graph
            target_features: Target features to activate
            metric: Scoring method

        Returns:
            Score (higher is better)
        """
        active_set = set(tuple(f.tolist()) for f in graph.active_features)

        if metric == "activation_strength":
            if not target_features:
                # Use total activation if no targets specified
                return graph.activation_values.sum().item()
            return self._total_activation(graph, target_features)

        elif metric == "feature_coverage":
            if not target_features:
                # Return number of features activated
                return len(active_set)
            target_set = set(target_features)
            overlap = len(active_set & target_set)
            return overlap / len(target_set) if target_set else 0

        elif metric == "coherence":
            return compute_pathway_coherence(graph)

        elif metric == "balanced":
            coherence = compute_pathway_coherence(graph)

            if target_features:
                target_set = set(target_features)
                coverage = len(active_set & target_set) / len(target_set) if target_set else 0
                return 0.5 * coverage + 0.5 * coherence
            else:
                # No targets: balance feature count and coherence
                normalized_features = min(1.0, len(active_set) / 100)  # Cap at 100 features
                return 0.5 * normalized_features + 0.5 * coherence

        else:
            raise ValueError(f"Unknown metric: {metric}")

    def _total_activation(self, graph: Graph, target_features: list[tuple] | None) -> float:
        """Sum activation strength for target features.

        Args:
            graph: Attribution graph
            target_features: Features to sum activations for

        Returns:
            Total activation strength
        """
        if not target_features:
            return 0.0

        active_dict = {
            tuple(graph.active_features[i].tolist()): graph.activation_values[i].item()
            for i in range(len(graph.active_features))
        }

        total = sum(active_dict.get(feat, 0) for feat in target_features)
        return total

    def compare_phrasings(
        self,
        phrases: list[str],
        phrase_names: list[str] | None = None,
        target_features: list[tuple] | None = None,
        metric: str = "balanced",
    ) -> dict[str, Any]:
        """Compare multiple phrasings side-by-side.

        Args:
            phrases: List of phrases to compare
            phrase_names: Optional names for each phrase
            target_features: Target features
            metric: Scoring metric

        Returns:
            Comparison results with rankings
        """
        if phrase_names is None:
            phrase_names = [f"Phrasing {i+1}" for i in range(len(phrases))]

        # Evaluate each
        results = []
        for phrase in phrases:
            graph = attribute(phrase, self.model, **self.attribution_kwargs)
            score = self._score_variant(graph, target_features, metric)

            results.append({
                "phrase": phrase,
                "score": score,
                "graph": graph,
            })

        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)

        return {
            "rankings": [
                {
                    "rank": i + 1,
                    "name": phrase_names[phrases.index(r["phrase"])],
                    "phrase": r["phrase"],
                    "score": r["score"],
                }
                for i, r in enumerate(results)
            ],
            "winner": phrase_names[phrases.index(results[0]["phrase"])],
            "detailed_results": results,
            "metric_used": metric,
        }


# Utility function for simple paraphrase generation
def create_simple_paraphrase_generator(llm_generate_func: Callable[[str], str]):
    """Create a paraphrase generator from an LLM generation function.

    Args:
        llm_generate_func: Function that takes a prompt and returns generated text

    Returns:
        Paraphrase generator compatible with CircuitGuidedPhraseOptimizer

    Example:
        >>> from transformers import pipeline
        >>> generator = pipeline("text-generation", model="gpt2")
        >>>
        >>> def my_llm_generate(prompt):
        ...     return generator(prompt, max_length=100)[0]["generated_text"]
        >>>
        >>> paraphrase_gen = create_simple_paraphrase_generator(my_llm_generate)
        >>> optimizer = CircuitGuidedPhraseOptimizer(model, paraphrase_gen)
    """

    def paraphrase_generator(text: str, n: int) -> list[str]:
        """Generate n paraphrases of text."""
        prompt = f"""Generate {n} paraphrases of the following text.

Original: "{text}"

Requirements:
- Preserve core meaning
- Vary sentence structure and word choice
- Make them natural and fluent

Output as a JSON list: ["variant1", "variant2", ...]
"""

        response = llm_generate_func(prompt)

        # Try to parse JSON
        try:
            # Extract JSON from response
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                variants = json.loads(json_str)
                return variants[:n]  # Return up to n variants
        except Exception as e:
            logger.warning(f"Failed to parse JSON paraphrases: {e}")

        # Fallback: split by newlines
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        # Filter out the original and non-paraphrase lines
        variants = [
            line for line in lines if len(line) > 10 and line != text and not line.startswith("Original:")
        ]

        return variants[:n]

    return paraphrase_generator


# Example: Manual paraphrase generator (for testing without LLM)
def manual_paraphrase_generator(text: str, n: int) -> list[str]:
    """Simple rule-based paraphrase generator for testing.

    This is a placeholder - in production, use an LLM-based generator.
    """
    # Simple transformations
    variants = []

    # Add "According to" prefix
    if "the" in text.lower() and "is" in text.lower():
        variants.append(f"According to the information, {text.lower()}")

    # Rephrase with "it is stated that"
    variants.append(f"It is stated that {text.lower()}")

    # Add "In summary"
    variants.append(f"In summary, {text.lower()}")

    # Passive voice conversion (simple)
    if "the" in text.lower():
        variants.append(f"The following is noted: {text}")

    # Add "Based on the data"
    variants.append(f"Based on the data, {text.lower()}")

    # Pad with slight variations
    while len(variants) < n:
        variants.append(f"{text} (variant {len(variants)})")

    return variants[:n]
