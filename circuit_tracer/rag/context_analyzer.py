"""RAG Circuit Analyzer: Evaluate context quality by feature activation."""

import logging
from typing import Any

import torch
from tqdm import tqdm

from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.evaluation.metrics import compute_pathway_coherence
from circuit_tracer.graph import Graph

logger = logging.getLogger(__name__)


class RAGCircuitAnalyzer:
    """Analyze how well retrieval contexts activate task-relevant features.

    This tool helps optimize RAG by evaluating contexts based on:
    1. Which features they activate (not just semantic similarity)
    2. Circuit coherence (how interpretable the processing is)
    3. Preservation/enhancement of query features

    Use cases:
    - Re-rank retrieved documents by circuit quality
    - Debug why retrieval isn't working
    - Find contexts the model can actually process well
    """

    def __init__(
        self,
        model: ReplacementModel,
        expected_features: list[tuple] | None = None,
        attribution_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize the RAG circuit analyzer.

        Args:
            model: ReplacementModel to analyze
            expected_features: List of (layer, pos, feat_idx) tuples for task features
            attribution_kwargs: Additional kwargs for attribute()
        """
        self.model = model
        self.expected_features = set(expected_features) if expected_features else None
        self.attribution_kwargs = attribution_kwargs or {}

    def analyze_context_quality(
        self,
        query: str,
        retrieved_docs: list[str],
        verbose: bool = False,
    ) -> list[dict[str, Any]]:
        """Analyze and rank contexts by circuit quality.

        Args:
            query: The user query
            retrieved_docs: List of retrieved document strings
            verbose: Whether to show progress

        Returns:
            List of context results, sorted by quality score (best first)
        """
        # Baseline: query alone
        logger.info("Computing baseline (query only)...")
        baseline_graph = attribute(query, self.model, **self.attribution_kwargs)
        baseline_features = set(tuple(f.tolist()) for f in baseline_graph.active_features)

        # Analyze each retrieved doc
        context_results = []
        iterator = tqdm(retrieved_docs, desc="Analyzing contexts") if verbose else retrieved_docs

        for doc in iterator:
            full_prompt = self._format_prompt(query, doc)
            context_graph = attribute(full_prompt, self.model, **self.attribution_kwargs)
            context_features = set(tuple(f.tolist()) for f in context_graph.active_features)

            # Compute metrics
            new_features = context_features - baseline_features
            lost_features = baseline_features - context_features

            quality_score = self._compute_quality(
                new_features=new_features,
                lost_features=lost_features,
                graph=context_graph,
            )

            context_results.append({
                "doc": doc,
                "quality_score": quality_score,
                "new_features_count": len(new_features),
                "lost_features_count": len(lost_features),
                "total_features": len(context_features),
                "coherence": compute_pathway_coherence(context_graph),
                "new_features": list(new_features)[:10],  # Top 10 for inspection
                "graph": context_graph,  # Full graph for detailed analysis
            })

        # Sort by quality score (best first)
        context_results.sort(key=lambda x: x["quality_score"], reverse=True)

        return context_results

    def compare_contexts(
        self,
        query: str,
        contexts: list[str],
        context_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare multiple contexts side-by-side.

        Args:
            query: The user query
            contexts: List of context strings to compare
            context_names: Optional names for each context

        Returns:
            Comparison dictionary with rankings and metrics
        """
        if context_names is None:
            context_names = [f"Context {i+1}" for i in range(len(contexts))]

        results = self.analyze_context_quality(query, contexts)

        comparison = {
            "query": query,
            "rankings": [
                {
                    "rank": i + 1,
                    "name": context_names[contexts.index(r["doc"])],
                    "quality_score": r["quality_score"],
                    "coherence": r["coherence"],
                    "new_features": r["new_features_count"],
                }
                for i, r in enumerate(results)
            ],
            "winner": context_names[contexts.index(results[0]["doc"])],
            "detailed_results": results,
        }

        return comparison

    def _format_prompt(self, query: str, context: str) -> str:
        """Format query and context into a prompt.

        Override this method to customize prompt formatting.
        """
        return f"Context: {context}\n\nQuestion: {query}"

    def _compute_quality(
        self,
        new_features: set,
        lost_features: set,
        graph: Graph,
    ) -> float:
        """Compute overall quality score for a context.

        Args:
            new_features: Features activated by context (not in baseline)
            lost_features: Query features lost when adding context
            graph: Full attribution graph

        Returns:
            Quality score (higher is better)
        """
        # Component scores
        coherence = compute_pathway_coherence(graph)
        new_feature_score = len(new_features)
        preservation_score = 1.0 / (1.0 + len(lost_features))  # Penalize lost features

        # Check expected feature activation (if provided)
        expected_score = 1.0
        if self.expected_features:
            active_set = set(tuple(f.tolist()) for f in graph.active_features)
            overlap = len(active_set & self.expected_features)
            expected_score = overlap / len(self.expected_features) if self.expected_features else 0

        # Weighted combination
        quality = (
            0.3 * new_feature_score +  # New information added
            0.3 * coherence +  # Circuit interpretability
            0.2 * preservation_score +  # Query features preserved
            0.2 * expected_score  # Task-relevant features activated
        )

        return quality

    def rerank_documents(
        self,
        query: str,
        retrieved_docs: list[str],
        top_k: int = 5,
        verbose: bool = False,
    ) -> list[str]:
        """Re-rank retrieved documents by circuit quality.

        Args:
            query: The user query
            retrieved_docs: List of retrieved documents
            top_k: Number of top documents to return
            verbose: Whether to show progress

        Returns:
            Top-k documents ranked by circuit quality
        """
        results = self.analyze_context_quality(query, retrieved_docs, verbose=verbose)

        return [r["doc"] for r in results[:top_k]]

    def debug_retrieval(
        self,
        query: str,
        retrieved_docs: list[str],
        min_quality_threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Debug retrieval by identifying poor-quality contexts.

        Args:
            query: The user query
            retrieved_docs: Retrieved documents
            min_quality_threshold: Minimum acceptable quality score

        Returns:
            Debug report with problem contexts and statistics
        """
        results = self.analyze_context_quality(query, retrieved_docs)

        poor_contexts = [r for r in results if r["quality_score"] < min_quality_threshold]
        good_contexts = [r for r in results if r["quality_score"] >= min_quality_threshold]

        report = {
            "query": query,
            "total_contexts": len(retrieved_docs),
            "good_contexts": len(good_contexts),
            "poor_contexts": len(poor_contexts),
            "avg_quality": sum(r["quality_score"] for r in results) / len(results),
            "avg_coherence": sum(r["coherence"] for r in results) / len(results),
            "problems": [
                {
                    "doc": r["doc"][:100] + "...",
                    "quality_score": r["quality_score"],
                    "coherence": r["coherence"],
                    "lost_features": r["lost_features_count"],
                    "diagnosis": self._diagnose_problem(r),
                }
                for r in poor_contexts[:5]  # Top 5 worst
            ],
            "recommendations": self._generate_recommendations(results),
        }

        return report

    def _diagnose_problem(self, result: dict) -> str:
        """Diagnose why a context has poor quality."""
        issues = []

        if result["coherence"] < 0.3:
            issues.append("Low coherence (diffuse processing)")
        if result["lost_features_count"] > 5:
            issues.append(f"Lost {result['lost_features_count']} query features")
        if result["new_features_count"] < 3:
            issues.append("Added few new features (low information)")
        if result["total_features"] > 500:
            issues.append("Too many features (overly complex)")

        return "; ".join(issues) if issues else "No clear issues"

    def _generate_recommendations(self, results: list[dict]) -> list[str]:
        """Generate recommendations based on results."""
        recommendations = []

        avg_coherence = sum(r["coherence"] for r in results) / len(results)
        avg_lost = sum(r["lost_features_count"] for r in results) / len(results)

        if avg_coherence < 0.4:
            recommendations.append(
                "Low average coherence - contexts may be too complex or poorly formatted"
            )

        if avg_lost > 3:
            recommendations.append(
                f"High feature loss (avg {avg_lost:.1f}) - contexts may be overriding query information"
            )

        if all(r["new_features_count"] < 5 for r in results):
            recommendations.append(
                "Few new features across all contexts - retrieval may not be finding relevant information"
            )

        if not recommendations:
            recommendations.append("Overall quality looks good!")

        return recommendations
