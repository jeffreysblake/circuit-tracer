# Novel Research Applications: Circuit-Informed RAG & Agent Optimization

**Status**: Research Ideas → Implementation

This document describes novel applications of circuit tracing and sparse autoencoders (SAEs) for improving Retrieval-Augmented Generation (RAG) and agent systems. These ideas are not extensively researched in current literature and represent opportunities for innovation.

---

## Core Insight

**Traditional RAG Problem**: We retrieve documents based on semantic similarity, but don't know:
- Whether the model actually *uses* the retrieved context
- Which phrasings activate task-relevant features best
- If poor performance is due to retrieval or processing

**Our Solution**: Use circuit analysis to:
1. Evaluate how well contexts activate task features
2. Optimize phrasings for better feature activation
3. Validate semantic clustering with circuit metrics
4. Discover which features matter for specific tasks

---

## Novel Application 1: Context Feature Profiling

### Concept

**Problem**: Retrieved documents may be semantically similar but poorly processed by the model.

**Solution**: Analyze which contexts best activate task-relevant features.

### Implementation

```python
class RAGCircuitAnalyzer:
    """Analyze how well contexts activate task-relevant features."""

    def analyze_context_quality(
        self,
        query: str,
        retrieved_docs: list[str],
        expected_features: list[str],  # Expected feature patterns
    ):
        """Compare feature activation with/without context."""

        # Baseline: query alone
        baseline_graph = attribute(query, self.model)
        baseline_features = set(tuple(f.tolist()) for f in baseline_graph.active_features)

        # Analyze each retrieved doc
        context_results = []
        for doc in retrieved_docs:
            full_prompt = f"Context: {doc}\n\nQuestion: {query}"
            context_graph = attribute(full_prompt, self.model)
            context_features = set(tuple(f.tolist()) for f in context_graph.active_features)

            # Metrics
            new_features = context_features - baseline_features
            lost_features = baseline_features - context_features

            # Score based on:
            # 1. New task-relevant features activated
            # 2. Coherence of resulting circuit
            # 3. Preservation of query features

            quality_score = self._compute_quality(
                new_features=new_features,
                lost_features=lost_features,
                coherence=compute_pathway_coherence(context_graph),
                expected_patterns=expected_features,
            )

            context_results.append({
                "doc": doc,
                "quality_score": quality_score,
                "new_features": len(new_features),
                "lost_features": len(lost_features),
                "coherence": compute_pathway_coherence(context_graph),
            })

        # Re-rank by circuit quality
        return sorted(context_results, key=lambda x: x["quality_score"], reverse=True)
```

### Use Cases

**1. Context Re-Ranking**
```python
# Retrieve 20 documents
retrieved = retriever.search(query, k=20)

# Analyze with circuits
analyzer = RAGCircuitAnalyzer(model, task_config)
ranked = analyzer.analyze_context_quality(query, retrieved, expected_features)

# Use top 3 by circuit quality (not just semantic similarity)
best_contexts = [r["doc"] for r in ranked[:3]]
final_answer = model.generate(f"Context: {best_contexts}\n\n{query}")
```

**2. Retrieval Debugging**
```python
# Find why retrieval isn't working
results = analyzer.analyze_context_quality(query, retrieved, expected_features)

# Check if problem is retrieval or processing
for r in results:
    if r["quality_score"] < 0.3:
        print(f"⚠️ Poor processing: {r['doc'][:50]}...")
        print(f"   New features: {r['new_features']}")
        print(f"   Coherence: {r['coherence']:.2f}")
```

### Why This Is Novel

- **Current RAG**: Uses only semantic similarity (embeddings)
- **Our approach**: Adds *processing quality* (circuit analysis)
- **Benefit**: Find contexts the model can actually use, not just similar ones

---

## Novel Application 2: Phrase Optimization via Circuit Feedback

### Concept

**Problem**: Subtle phrasing differences can dramatically affect model behavior, but we optimize via trial-and-error.

**Solution**: Use circuit analysis as an objective function to optimize phrasings.

### Implementation

```python
class CircuitGuidedPhraseOptimizer:
    """Optimize phrases to maximize task-relevant feature activation."""

    def optimize_phrase(
        self,
        original_phrase: str,
        target_features: list[tuple],  # (layer, pos, feat_idx) to activate
        n_variants: int = 5,
        optimization_metric: str = "activation_strength",
    ):
        """Generate variants and select best by circuit activation."""

        # Step 1: Generate paraphrases (LLM-based)
        variants = self._generate_paraphrases(original_phrase, n_variants)
        variants.append(original_phrase)  # Include original

        # Step 2: Evaluate each with circuit analysis
        results = []
        for variant in variants:
            graph = attribute(variant, self.model)

            # Score based on target feature activation
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
                "activation_strength": self._total_activation(graph, target_features),
            })

        # Step 3: Return best variant
        best = max(results, key=lambda x: x["score"])

        return {
            "optimized": best["phrase"],
            "original": original_phrase,
            "improvement": best["score"] / results[-1]["score"],  # vs original
            "all_variants": results,
        }

    def _generate_paraphrases(self, text: str, n: int):
        """Use LLM to generate paraphrases."""
        prompt = f"""Generate {n} paraphrases of the following text.

Original: "{text}"

Requirements:
- Preserve core meaning
- Vary sentence structure and word choice
- Make them natural and fluent

Output as a JSON list: ["variant1", "variant2", ...]
"""
        response = self.llm_generator.generate(prompt)
        return json.loads(response)

    def _score_variant(self, graph, target_features, metric):
        """Score a variant based on circuit properties."""
        active_set = set(tuple(f.tolist()) for f in graph.active_features)
        target_set = set(target_features)

        if metric == "activation_strength":
            # Sum of activation values for target features
            return self._total_activation(graph, target_features)

        elif metric == "feature_coverage":
            # % of target features activated
            overlap = len(active_set & target_set)
            return overlap / len(target_set) if target_set else 0

        elif metric == "balanced":
            # Combine coverage and coherence
            coverage = len(active_set & target_set) / len(target_set) if target_set else 0
            coherence = compute_pathway_coherence(graph)
            return 0.6 * coverage + 0.4 * coherence

    def _total_activation(self, graph, target_features):
        """Sum activation strength for target features."""
        active_dict = {
            tuple(graph.active_features[i].tolist()): graph.activation_values[i].item()
            for i in range(len(graph.active_features))
        }

        total = sum(active_dict.get(feat, 0) for feat in target_features)
        return total
```

### Use Cases

**1. Optimize RAG Context Formatting**
```python
# Original context
original = "The meeting is scheduled for January 5th, 2025 at 3pm."

# Discover features for date extraction
date_features = discover_task_features(
    task="date_extraction",
    examples=[...],
)

# Optimize phrasing
optimizer = CircuitGuidedPhraseOptimizer(model, llm_generator)
result = optimizer.optimize_phrase(
    original,
    target_features=date_features,
)

print(f"Original: {original}")
print(f"Optimized: {result['optimized']}")
print(f"Improvement: {result['improvement']:.2f}x")

# Use optimized version in RAG
# Output might be: "Date: 2025-01-05 (January 5, 2025) | Time: 15:00"
```

**2. Optimize Agent Prompts**
```python
# Tool calling prompt
original_tool_prompt = "Use the weather function to check Paris"

# Discover tool-calling features
tool_features = discover_task_features(
    task="tool_calling",
    examples=[...],
)

# Optimize
result = optimizer.optimize_phrase(original_tool_prompt, tool_features)

# Optimized might be: "Call get_weather(location='Paris')"
# Which better activates function-calling features
```

**3. A/B Testing with Objective Metrics**
```python
variants = [
    "According to the document...",
    "Based on the provided information...",
    "The text states that...",
]

# Test each
for v in variants:
    graph = attribute(v + " " + query, model)
    score = compute_feature_coverage(graph, rag_features)
    print(f"{v}: {score:.2f}")

# Use the winner systematically
```

### Why This Is Novel

- **Current approach**: A/B test with task performance (slow, expensive)
- **Our approach**: Use circuits as objective function (fast, cheap)
- **Benefit**: Optimize prompts based on *how the model processes them*, not just outcomes

---

## Novel Application 3: Semantic Grouping + Circuit Validation

### Concept

**Problem**: Semantic clustering groups similar documents, but similarity ≠ processing quality.

**Solution**: Cluster semantically, then validate clusters with circuit analysis.

### Implementation

```python
class SemanticCircuitValidator:
    """Group documents semantically, validate processing quality with circuits."""

    def cluster_and_validate(
        self,
        documents: list[str],
        query: str,
        n_clusters: int = 5,
        task_features: list[tuple] | None = None,
    ):
        """Cluster docs by embeddings, pick best cluster via circuits."""

        # Step 1: Semantic clustering
        embeddings = self.embedder.encode(documents)
        cluster_labels = KMeans(n_clusters=n_clusters).fit_predict(embeddings)

        # Step 2: For each cluster, evaluate with circuits
        cluster_scores = {}
        for cluster_id in range(n_clusters):
            cluster_docs = [d for d, c in zip(documents, cluster_labels) if c == cluster_id]

            # Use centroid or representative doc
            representative = self._get_representative(cluster_docs, embeddings[cluster_labels == cluster_id])

            # Circuit analysis
            prompt = f"Context: {representative}\n\nQuestion: {query}"
            graph = attribute(prompt, self.model)

            # Score by circuit metrics
            coverage = compute_feature_coverage(graph, task_features) if task_features else 0
            coherence = compute_pathway_coherence(graph)
            density = compute_graph_density(graph)

            # Combined score
            circuit_score = (
                0.4 * coverage +
                0.4 * coherence +
                0.2 * (1 - density)  # Prefer sparse (focused) graphs
            )

            cluster_scores[cluster_id] = {
                "circuit_score": circuit_score,
                "docs": cluster_docs,
                "metrics": {
                    "coverage": coverage,
                    "coherence": coherence,
                    "density": density,
                },
                "representative": representative,
            }

        # Step 3: Return best cluster
        best_id = max(cluster_scores.keys(), key=lambda k: cluster_scores[k]["circuit_score"])

        return cluster_scores[best_id], cluster_scores

    def _get_representative(self, docs, embeddings):
        """Find document closest to cluster centroid."""
        centroid = embeddings.mean(axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        closest_idx = distances.argmin()
        return docs[closest_idx]
```

### Use Cases

**1. Intelligent Document Grouping**
```python
# Retrieve 100 documents
retrieved = retriever.search(query, k=100)

# Cluster and validate
validator = SemanticCircuitValidator(model, embedder)
best_cluster, all_clusters = validator.cluster_and_validate(
    documents=retrieved,
    query=query,
    n_clusters=5,
    task_features=task_features,
)

print(f"Best cluster: {len(best_cluster['docs'])} docs")
print(f"Circuit score: {best_cluster['circuit_score']:.2f}")

# Use only the best cluster
final_context = "\n\n".join(best_cluster["docs"][:3])
```

**2. Multi-Aspect Retrieval**
```python
# Query requires multiple aspects (e.g., "Compare X and Y")
aspects = ["aspect_X", "aspect_Y"]

results = {}
for aspect in aspects:
    aspect_query = f"{query} - focusing on {aspect}"
    best, all = validator.cluster_and_validate(retrieved, aspect_query, n_clusters=3)
    results[aspect] = best["docs"][0]  # Best doc for this aspect

# Combine aspects
final_context = f"Regarding X: {results['aspect_X']}\n\nRegarding Y: {results['aspect_Y']}"
```

### Why This Is Novel

- **Semantic clustering**: Groups by similarity
- **Circuit validation**: Ensures model can process the group
- **Benefit**: Best of both worlds - semantic coherence + processing quality

---

## Novel Application 4: Task Feature Discovery

### Concept

**Problem**: We need to know which features are relevant for a task, but manual discovery is tedious.

**Solution**: Automatically discover task-critical features by comparing task vs baseline prompts.

### Implementation

```python
class TaskFeatureDiscovery:
    """Automatically discover features critical for a specific task."""

    def discover_critical_features(
        self,
        task_prompts: list[str],
        baseline_prompts: list[str],
        top_k: int = 20,
    ):
        """Find features that activate strongly on task but not baseline."""

        # Run attribution on all prompts
        task_graphs = [attribute(p, self.model) for p in task_prompts]
        baseline_graphs = [attribute(p, self.model) for p in baseline_prompts]

        # Count feature activations
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
            baseline_count = baseline_feature_counts.get(feat, 0)

            # Task-specificity score
            # High if: frequent in task, rare in baseline, strong activation
            task_freq = task_count / len(task_prompts)
            baseline_freq = baseline_count / len(baseline_prompts) if baseline_count > 0 else 0.01
            specificity = task_freq / baseline_freq
            strength = task_feature_strength[feat] / task_count

            feature_scores[feat] = specificity * strength

        # Return top-k task-critical features
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

        return {
            "features": [f[0] for f in sorted_features[:top_k]],
            "scores": [f[1] for f in sorted_features[:top_k]],
            "task_frequency": {f: task_feature_counts[f] / len(task_prompts) for f, _ in sorted_features[:top_k]},
            "baseline_frequency": {f: baseline_feature_counts.get(f, 0) / len(baseline_prompts) for f, _ in sorted_features[:top_k]},
        }

    def create_feature_signature(
        self,
        task_name: str,
        critical_features: list[tuple],
    ):
        """Create a reusable signature for this task."""
        return {
            "task_name": task_name,
            "features": critical_features,
            "created_at": datetime.now().isoformat(),
        }
```

### Use Cases

**1. Automatic Task Configuration**
```python
# Define task by examples
tool_use_examples = [
    "Use get_weather to check Paris",
    "Call the search function for 'AI'",
    "Invoke calculate(5, '+', 3)",
]

general_examples = [
    "What is the weather?",
    "Tell me about AI",
    "What is 5 + 3?",
]

# Discover features
discoverer = TaskFeatureDiscovery(model)
result = discoverer.discover_critical_features(
    task_prompts=tool_use_examples,
    baseline_prompts=general_examples,
    top_k=15,
)

print(f"Top tool-use features: {result['features']}")
print(f"Task frequency: {result['task_frequency']}")

# Save as task signature
signature = discoverer.create_feature_signature("tool_use", result["features"])

# Use for evaluation
task_config = load_task_config("task_configs/tool_use.yaml")
task_config.expected_features = signature["features"]
```

**2. Monitor Fine-Tuning**
```python
# Before fine-tuning
initial_features = discover_critical_features(task_prompts, baseline_prompts)

# After fine-tuning
final_features = discover_critical_features(task_prompts, baseline_prompts)

# Compare
new_features = set(final_features) - set(initial_features)
lost_features = set(initial_features) - set(final_features)

print(f"Learned {len(new_features)} new task features")
print(f"Lost {len(lost_features)} initial features")
```

### Why This Is Novel

- **Manual approach**: Inspect graphs, label features by hand
- **Our approach**: Automated discovery via differential analysis
- **Benefit**: Scale feature discovery to many tasks

---

## Comparison to Existing Work

### Goodfire's Feature Steering (Jan 2025)

**What they do**:
- Identify interpretable features via SAEs
- Provide API to amplify/suppress features at inference
- Focus on behavior modification (tone, style, refusal)

**What we add**:
- Use circuits for *optimization* not just steering
- Focus on RAG and agent use cases
- Automatic feature discovery
- Quality metrics (coherence, coverage) not just activation

**Complementary**: Could use Goodfire for steering, our tools for optimization

### Anthropic's Attribution Graphs

**What they do**:
- Visualize feature interactions
- Understand model reasoning
- Research tool for interpretability

**What we add**:
- *Apply* to practical problems (RAG, prompt optimization)
- Automated workflows (clustering, re-ranking, discovery)
- Quantitative metrics for optimization

---

## Research Questions & Future Work

### Open Questions

1. **Feature stability**: Do discovered features transfer across models?
2. **Optimization convergence**: How many variants needed to find optimal phrasing?
3. **Circuit-performance correlation**: Do our metrics predict downstream task performance?
4. **Computational cost**: Can we approximate circuits for real-time re-ranking?

### Next Steps

1. **Empirical validation**: Test on standard RAG benchmarks (MS MARCO, Natural Questions)
2. **Ablation studies**: Which metrics matter most? (coverage vs coherence vs density)
3. **Comparison baselines**: Circuit-based vs embedding-based re-ranking
4. **Scaling**: Can we cache feature profiles for common queries?

### Potential Publications

- "Circuit-Informed Retrieval: Using Mechanistic Interpretability for RAG Optimization"
- "Phrase Optimization via Sparse Autoencoder Features"
- "Semantic Clustering with Circuit Validation"

---

## Implementation Roadmap

**Phase 1** (Current): Core implementations
- [x] RAGCircuitAnalyzer
- [x] CircuitGuidedPhraseOptimizer
- [x] SemanticCircuitValidator
- [x] TaskFeatureDiscovery

**Phase 2**: Evaluation & Benchmarking
- [ ] Test on MS MARCO / Natural Questions
- [ ] Compare to baseline re-rankers
- [ ] Measure computational cost
- [ ] Ablation studies

**Phase 3**: Integration & UX
- [ ] HuggingFace Datasets integration
- [ ] LangChain/LlamaIndex plugins
- [ ] CLI tools for common workflows
- [ ] Visualization dashboards

**Phase 4**: Research & Publication
- [ ] Write paper on circuit-informed RAG
- [ ] Open-source benchmark suite
- [ ] Case studies on real applications

---

## Key Insights

1. **Circuits reveal processing, not just similarity**: Semantic search finds similar docs, circuits show if the model can use them.

2. **Optimization needs objective functions**: Circuit metrics provide fast, cheap feedback for prompt/context optimization.

3. **Features are compositional**: Task-critical features can be discovered, reused, and transferred.

4. **Interpretability enables engineering**: Moving from "understanding models" to "using understanding to build better systems".

---

## Contact & Collaboration

This is novel research territory. If you're interested in:
- Collaborating on implementations
- Running experiments on your RAG system
- Contributing to benchmarks
- Co-authoring papers

These ideas could form the basis for significant research contributions in mechanistic interpretability applied to practical systems.

**Last Updated**: 2025-11-16
**Status**: Active Research & Implementation
