"""Semantic Circuit Validator: Group documents semantically, validate with circuits."""

import logging
from typing import Any

import numpy as np
import torch

from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.evaluation.metrics import (
    compute_feature_coverage,
    compute_graph_density,
    compute_pathway_coherence,
)

logger = logging.getLogger(__name__)


class SemanticCircuitValidator:
    """Group documents semantically, validate processing quality with circuits.

    Combines semantic clustering (embeddings) with circuit validation to find
    document groups that are both semantically coherent AND well-processed.

    Use cases:
    - Intelligent document grouping for RAG
    - Multi-aspect retrieval (find best docs for each aspect)
    - Validate that semantic similarity → processing quality
    """

    def __init__(
        self,
        model: ReplacementModel,
        embedder: Any = None,  # sentence_transformers.SentenceTransformer or similar
        attribution_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize the semantic circuit validator.

        Args:
            model: ReplacementModel for circuit analysis
            embedder: Embedding model (e.g., SentenceTransformer)
                     If None, must provide embeddings to cluster_and_validate
            attribution_kwargs: Additional kwargs for attribute()
        """
        self.model = model
        self.embedder = embedder
        self.attribution_kwargs = attribution_kwargs or {}

    def cluster_and_validate(
        self,
        documents: list[str],
        query: str,
        n_clusters: int = 5,
        task_features: list[tuple] | None = None,
        embeddings: np.ndarray | None = None,
        clustering_method: str = "kmeans",
        verbose: bool = False,
    ) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
        """Cluster docs by embeddings, pick best cluster via circuits.

        Args:
            documents: List of document strings
            query: User query for context
            n_clusters: Number of clusters
            task_features: Optional task-relevant features
            embeddings: Pre-computed embeddings, or None to compute
            clustering_method: "kmeans" or "agglomerative"
            verbose: Whether to show progress

        Returns:
            Tuple of (best_cluster_info, all_cluster_scores)
        """
        # Compute embeddings if not provided
        if embeddings is None:
            if self.embedder is None:
                raise ValueError("Must provide embeddings or embedder")
            if verbose:
                logger.info("Computing embeddings...")
            embeddings = self.embedder.encode(documents, show_progress_bar=verbose)

        # Cluster
        if verbose:
            logger.info(f"Clustering into {n_clusters} groups...")

        cluster_labels = self._cluster(embeddings, n_clusters, clustering_method)

        # Evaluate each cluster with circuits
        if verbose:
            logger.info("Evaluating clusters with circuit analysis...")

        cluster_scores = {}
        for cluster_id in range(n_clusters):
            cluster_docs = [d for d, c in zip(documents, cluster_labels) if c == cluster_id]

            if not cluster_docs:
                continue

            # Get representative document
            cluster_embeddings = embeddings[cluster_labels == cluster_id]
            representative = self._get_representative(cluster_docs, cluster_embeddings)

            # Circuit analysis
            prompt = self._format_prompt(query, representative)
            graph = attribute(prompt, self.model, **self.attribution_kwargs)

            # Compute scores
            coverage = (
                compute_feature_coverage(graph, [f.__dict__ if hasattr(f, "__dict__") else f for f in task_features])
                if task_features
                else 0
            )
            coherence = compute_pathway_coherence(graph)
            density = compute_graph_density(graph)

            # Combined circuit score
            circuit_score = (
                0.4 * coverage + 0.4 * coherence + 0.2 * (1 - density)  # Prefer sparse graphs
            )

            cluster_scores[cluster_id] = {
                "circuit_score": circuit_score,
                "docs": cluster_docs,
                "doc_count": len(cluster_docs),
                "metrics": {
                    "coverage": coverage,
                    "coherence": coherence,
                    "density": density,
                },
                "representative": representative,
                "graph": graph,
            }

        # Find best cluster
        if not cluster_scores:
            raise ValueError("No valid clusters found")

        best_cluster_id = max(
            cluster_scores.keys(), key=lambda k: cluster_scores[k]["circuit_score"]
        )

        return cluster_scores[best_cluster_id], cluster_scores

    def _cluster(
        self, embeddings: np.ndarray, n_clusters: int, method: str
    ) -> np.ndarray:
        """Cluster embeddings.

        Args:
            embeddings: Document embeddings
            n_clusters: Number of clusters
            method: Clustering method

        Returns:
            Cluster labels
        """
        try:
            from sklearn.cluster import AgglomerativeClustering, KMeans
        except ImportError:
            raise ImportError("scikit-learn required for clustering. Install with: pip install scikit-learn")

        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == "agglomerative":
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        labels = clusterer.fit_predict(embeddings)
        return labels

    def _get_representative(
        self, docs: list[str], embeddings: np.ndarray
    ) -> str:
        """Find document closest to cluster centroid.

        Args:
            docs: Documents in cluster
            embeddings: Embeddings for docs

        Returns:
            Representative document
        """
        if len(docs) == 1:
            return docs[0]

        # Find centroid
        centroid = embeddings.mean(axis=0)

        # Find closest document
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        closest_idx = distances.argmin()

        return docs[closest_idx]

    def _format_prompt(self, query: str, context: str) -> str:
        """Format query and context into prompt."""
        return f"Context: {context}\n\nQuestion: {query}"

    def multi_aspect_retrieval(
        self,
        documents: list[str],
        query: str,
        aspects: list[str],
        embeddings: np.ndarray | None = None,
        n_clusters_per_aspect: int = 3,
    ) -> dict[str, dict[str, Any]]:
        """Retrieve best documents for multiple query aspects.

        Args:
            documents: All documents
            query: Base query
            aspects: List of aspects (e.g., ["technical details", "business impact"])
            embeddings: Pre-computed embeddings
            n_clusters_per_aspect: Clusters to create per aspect

        Returns:
            Dictionary mapping aspects to their best documents
        """
        results = {}

        for aspect in aspects:
            aspect_query = f"{query} - focusing on {aspect}"

            best_cluster, all_clusters = self.cluster_and_validate(
                documents=documents,
                query=aspect_query,
                n_clusters=n_clusters_per_aspect,
                embeddings=embeddings,
            )

            results[aspect] = {
                "best_doc": best_cluster["docs"][0],
                "circuit_score": best_cluster["circuit_score"],
                "metrics": best_cluster["metrics"],
                "all_docs": best_cluster["docs"],
            }

        return results

    def validate_semantic_similarity(
        self,
        doc1: str,
        doc2: str,
        query: str,
        embedding_similarity: float | None = None,
    ) -> dict[str, Any]:
        """Test if semantic similarity predicts circuit similarity.

        Args:
            doc1: First document
            doc2: Second document
            query: Query for context
            embedding_similarity: Pre-computed cosine similarity, or None

        Returns:
            Comparison of semantic vs circuit similarity
        """
        # Semantic similarity
        if embedding_similarity is None and self.embedder:
            emb1 = self.embedder.encode([doc1])[0]
            emb2 = self.embedder.encode([doc2])[0]
            embedding_similarity = float(
                np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            )

        # Circuit analysis
        graph1 = attribute(
            self._format_prompt(query, doc1), self.model, **self.attribution_kwargs
        )
        graph2 = attribute(
            self._format_prompt(query, doc2), self.model, **self.attribution_kwargs
        )

        # Circuit similarity (Jaccard on features)
        features1 = set(tuple(f.tolist()) for f in graph1.active_features)
        features2 = set(tuple(f.tolist()) for f in graph2.active_features)

        intersection = len(features1 & features2)
        union = len(features1 | features2)
        circuit_similarity = intersection / union if union > 0 else 0

        # Coherence similarity
        coherence1 = compute_pathway_coherence(graph1)
        coherence2 = compute_pathway_coherence(graph2)
        coherence_similarity = 1 - abs(coherence1 - coherence2)

        return {
            "embedding_similarity": embedding_similarity,
            "circuit_similarity": circuit_similarity,
            "coherence_similarity": coherence_similarity,
            "correlation": circuit_similarity / embedding_similarity
            if embedding_similarity and embedding_similarity > 0
            else None,
            "doc1_coherence": coherence1,
            "doc2_coherence": coherence2,
        }
