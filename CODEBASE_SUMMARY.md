# Circuit-Tracer: Codebase Summary & Recommendations

## Executive Summary

Circuit-tracer is a **mechanistic interpretability** library that reveals *why* LLMs generate specific outputs for given inputs. It creates **attribution graphs** - visual representations of the computational pathways models use to determine outputs. This enables researchers to understand model reasoning by tracing causal relationships between interpretable features discovered through sparse coding techniques.

**Key Innovation**: The library replaces opaque MLP layers with **transcoders** - sparse autoencoders that decompose neural activations into interpretable features. These features become nodes in attribution graphs, with edges representing causal influence between features across layers.

## What This Repository Does

### Core Functionality

1. **Circuit Discovery via Attribution Graphs**
   - Computes direct effects between transcoder features, error nodes, input tokens, and output logits
   - Traces computational pathways showing *how* and *why* specific outputs are generated
   - Identifies which features causally influence others across transformer layers

2. **Visualization & Annotation**
   - Interactive web interface (local server or Neuronpedia integration)
   - Graph pruning to focus on most influential features
   - Manual annotation of features and feature grouping
   - Pin/unpin nodes, create supernodes, edit labels

3. **Feature Interventions**
   - Set transcoder features to arbitrary values
   - Observe downstream effects on model outputs
   - Validate hypotheses about feature causality
   - Test model behavior under controlled manipulations

### Technical Approach

The library implements the methodology from two 2025 Anthropic papers:
- [Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) (Methods)
- [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) (Applications)

**Attribution Algorithm**:
1. Create a **local replacement model** where gradients flow only through linear components (bypassing attention, MLP nonlinearities, LayerNorm scales)
2. **Forward pass**: Record residual stream activations and identify active features
3. **Backward passes**: For each source node, inject custom gradients selecting encoder/decoder directions
4. **Assemble graph**: Store edge weights showing direct effects A_{s→t} between features

## Repository Structure

```
circuit-tracer/
├── circuit_tracer/
│   ├── __main__.py              # CLI entry point
│   ├── replacement_model.py     # ReplacementModel with transcoder hooks
│   ├── graph.py                 # Graph data structure (adjacency matrix)
│   ├── attribution/
│   │   ├── attribute.py         # Core attribution algorithm
│   │   └── context.py           # Attribution context management
│   ├── transcoder/
│   │   ├── single_layer_transcoder.py   # Per-Layer Transcoders (PLT)
│   │   ├── cross_layer_transcoder.py    # Cross-Layer Transcoders (CLT)
│   │   └── activation_functions.py      # ReLU, JumpReLU, etc.
│   ├── frontend/
│   │   ├── local_server.py      # Local visualization server
│   │   ├── graph_models.py      # Pydantic models for graph JSON
│   │   └── feature_models.py    # Feature metadata models
│   └── utils/
│       ├── hf_utils.py           # HuggingFace transcoder loading
│       ├── create_graph_files.py # Graph pruning & JSON export
│       └── disk_offload.py       # Memory management
├── demos/                        # Jupyter notebooks & tutorials
└── tests/                        # Unit & integration tests
```

## Currently Supported Models

The library **requires pre-trained transcoders** to function. Currently available:

| Model | Transcoder Type | HuggingFace Repo | Notes |
|-------|----------------|------------------|-------|
| **Gemma-2 (2B)** | PLT (GemmaScope) | `mntss/gemma-scope-transcoders` | Default, shortcut: `gemma` |
| **Gemma-2 (2B)** | CLT (426K features) | `mntss/clt-gemma-2-2b-426k` | Cross-layer transcoders |
| **Gemma-2 (2B)** | CLT (2.5M features) | `mntss/clt-gemma-2-2b-2.5M` | Higher feature count |
| **Llama-3.2 (1B)** | PLT | `mntss/transcoder-Llama-3.2-1B` | Shortcut: `llama` |
| **Llama-3.2 (1B)** | CLT (524K features) | `mntss/clt-llama-3.2-1b-524k` | Cross-layer variant |
| **Qwen-3 (0.6B-14B)** | PLT | `mwhanna/qwen3-{size}-transcoders*` | Multiple sizes available |

## Can This Work With ANY Model?

### Short Answer: **Not out-of-the-box, but YES with transcoder training**

The circuit-tracer library itself is **model-agnostic** in architecture - it works with any transformer model supported by TransformerLens. However, it **requires pre-trained transcoders** specific to your target model.

### What You Need to Use Circuit-Tracer on a New Model

1. **The target LLM** (accessible via HuggingFace or TransformerLens)
2. **Pre-trained transcoders** for that model's MLP layers
3. **Computational resources** to run attribution (15GB+ GPU for small models)

### How to Train Transcoders for New Models

To use circuit-tracer on models like Claude, GPT-4, Mistral, etc., you must first train transcoders:

#### Recommended Libraries (2025):

1. **[EleutherAI Sparsify](https://github.com/EleutherAI/sparsify)** - Most streamlined
   ```bash
   # Train transcoders on any HuggingFace model
   python -m sparsify EleutherAI/pythia-160m [dataset] --transcode
   ```
   - Supports k-sparse autoencoders and transcoders
   - Command-line interface for easy training
   - Follows Gao et al. 2024 recipe

2. **[SAELens](https://github.com/jbloomAus/SAELens)** - Most comprehensive ecosystem
   - Extensive documentation for training custom SAEs
   - Pre-trained SAEs for many models
   - Active community and well-maintained

3. **[AI Safety Foundation sparse_autoencoder](https://github.com/ai-safety-foundation/sparse_autoencoder)**
   - Full PyTorch components for customization
   - Detailed demo notebook (`docs/content/demo.ipynb`)
   - Good for research and experimentation

#### Training Considerations:

**Transcoders vs SAEs**:
- **Transcoders** (recommended): Learn sparse decomposition of MLP input→output transformation
- **Sparse Autoencoders (SAEs)**: Decompose activations at a single point
- Recent research (Jan 2025) shows **transcoders produce more interpretable features**

**Resources Required**:
- Training dataset: Typically 100M-1B tokens from model's training distribution
- GPU: A100 (40GB+) or equivalent for medium models
- Time: Hours to days depending on model size
- Hyperparameters: Sparsity coefficient, learning rate, feature width (typically 4-32x model dimension)

**Key Hyperparameters**:
- `d_transcoder`: Number of features (e.g., 16K-2.5M)
- Sparsity penalty: L1 coefficient encouraging sparse activations
- Reconstruction loss: Ensure transcoders faithfully approximate MLP behavior

## Recommendations for Your Use Case

### Goal: Evaluate models for different tasks (tool use, RAG, etc.) by analyzing feature fitness

You want to:
1. Trace circuits in **multiple models** (ideally any model)
2. Evaluate how well existing feature patterns match **task-specific inputs** (tool calls, retrieval contexts, etc.)
3. Make this **easy to use locally** with a UI for model/dataset selection

### Architecture Recommendations

#### Phase 1: Foundation (Immediate - 2-4 weeks)

**1. Transcoder Training Pipeline**

Build an automated pipeline for training transcoders on new models:

```python
# Pseudocode for transcoder training wrapper
class TranscoderTrainer:
    def __init__(self, model_name: str, dataset: str):
        self.model = load_model(model_name)
        self.dataset = load_dataset(dataset)

    def train(self,
              d_transcoder: int = 65536,
              sparsity_coef: float = 0.01,
              n_tokens: int = 100_000_000):
        """Train transcoders using EleutherAI Sparsify or SAELens"""
        # Use existing libraries as backend
        pass

    def export_for_circuit_tracer(self, output_path: str):
        """Export in circuit-tracer compatible format"""
        # Create config.yaml with model_kind, hooks, etc.
        pass
```

**Recommended approach**:
- Use **EleutherAI Sparsify** as backend (simplest CLI interface)
- Create wrapper scripts that:
  - Accept model name + dataset
  - Train transcoders with sensible defaults
  - Export to HuggingFace format compatible with circuit-tracer
  - Generate `config.yaml` with proper hook names

**2. Task-Specific Dataset Curation**

For evaluating tool use, RAG, etc., create curated prompt sets:

```python
# Example dataset structure
task_datasets = {
    "tool_use": [
        {
            "prompt": "What's the weather in Paris? Use get_weather tool",
            "expected_features": ["tool_invocation", "location_extraction", "API_call"],
            "ground_truth_tool": "get_weather",
            "parameters": {"location": "Paris"}
        },
        # ... more examples
    ],
    "rag": [
        {
            "prompt": "According to the document, what is...?",
            "context": "...",
            "expected_features": ["context_reference", "information_extraction"],
            # ... metadata
        }
    ],
    # ... other tasks
}
```

**3. Evaluation Framework**

Create metrics to assess "feature fitness" for tasks:

```python
class CircuitEvaluator:
    def __init__(self, model: ReplacementModel, task: str):
        self.model = model
        self.task = task

    def evaluate_prompt(self, prompt: str):
        """Run attribution and compute task-specific metrics"""
        graph = attribute(prompt, self.model)

        return {
            "feature_coverage": self.compute_coverage(graph),
            "pathway_coherence": self.compute_coherence(graph),
            "task_specific_features": self.find_task_features(graph),
            "intervention_stability": self.test_interventions(graph)
        }

    def compute_coverage(self, graph):
        """What % of expected features are active?"""
        pass

    def compute_coherence(self, graph):
        """How connected/interpretable is the circuit?"""
        # E.g., graph density, path lengths, clustering coefficient
        pass

    def find_task_features(self, graph):
        """Identify features semantically related to the task"""
        # Could use feature descriptions, activation patterns, etc.
        pass

    def test_interventions(self, graph):
        """Test if manipulating features affects output predictably"""
        pass
```

#### Phase 2: UI Development (4-8 weeks)

**Recommended Tech Stack**:

**Option A: Web-Based UI (Recommended)**
- **Frontend**: React + TypeScript
  - Leverage existing [attribution-graphs-frontend](https://github.com/anthropics/attribution-graphs-frontend) as starting point
  - Add model selection, dataset management, batch processing
- **Backend**: FastAPI (Python)
  - Wrap circuit-tracer functionality
  - Manage transcoder training jobs
  - Store evaluation results
- **Database**: PostgreSQL or SQLite
  - Store graphs, annotations, evaluation metrics
- **Queue System**: Celery + Redis
  - Handle long-running attribution/training jobs

**Option B: Desktop UI (Simpler for local use)**
- **Framework**: Electron + React (cross-platform)
  - Or PyQt/PySide for pure Python solution
- **Architecture**: Similar to Option A but self-contained

**Key UI Features**:

1. **Model Management**
   ```
   [Model Selection]
   ├── Available Models
   │   ├── Gemma-2-2B ✓ (transcoders ready)
   │   ├── Llama-3.2-1B ✓
   │   ├── Mistral-7B ⚠ (needs transcoders)
   │   └── Custom Model [Add New]
   ├── Transcoder Status
   │   ├── Training Progress: 45%
   │   └── ETA: 2 hours
   └── Actions
       ├── [Train Transcoders]
       └── [Import Transcoders]
   ```

2. **Dataset Management**
   ```
   [Dataset Selection]
   ├── Built-in Tasks
   │   ├── Tool Use (234 prompts)
   │   ├── RAG (189 prompts)
   │   ├── Code Generation (456 prompts)
   │   └── Reasoning (312 prompts)
   ├── Custom Datasets
   │   └── [Upload CSV/JSON]
   └── Filters
       ├── Task Type: [All ▼]
       ├── Difficulty: [All ▼]
       └── Length: 0-500 tokens
   ```

3. **Batch Processing Dashboard**
   ```
   [Evaluation Run: tool-use-gemma-2b]
   Progress: ████████░░ 80% (40/50 prompts)

   Results Summary:
   ├── Avg Feature Coverage: 78%
   ├── Avg Pathway Coherence: 0.65
   ├── Task Features Found: 23/30 (77%)
   └── Intervention Stability: 0.82

   [View Graphs] [Export Results] [Compare Models]
   ```

4. **Interactive Attribution Graph**
   ```
   [Graph Viewer - Enhanced]
   ├── Standard features (from existing frontend)
   ├── Task-Specific Overlays
   │   ├── Highlight expected features
   │   ├── Show feature-task alignment scores
   │   └── Display intervention effects
   └── Comparison Mode
       └── Side-by-side graphs for different models
   ```

5. **Evaluation Analytics**
   ```
   [Model Comparison]

   Task: Tool Use
   ┌─────────────┬──────────┬──────────┬──────────┐
   │ Model       │ Coverage │ Coherence│ Stability│
   ├─────────────┼──────────┼──────────┼──────────┤
   │ Gemma-2-2B  │   78%    │   0.65   │   0.82   │
   │ Llama-3.2   │   82%    │   0.71   │   0.79   │
   │ Custom-7B   │   65%    │   0.58   │   0.74   │
   └─────────────┴──────────┴──────────┴──────────┘

   [Generate Report] [Export CSV]
   ```

**UI Architecture**:

```
┌─────────────────────────────────────────────────┐
│                  Web UI (React)                 │
│  ┌──────────┬──────────┬──────────┬──────────┐ │
│  │  Models  │ Datasets │  Graphs  │ Analytics│ │
│  └──────────┴──────────┴──────────┴──────────┘ │
└────────────────────┬────────────────────────────┘
                     │ REST API
┌────────────────────┴────────────────────────────┐
│            FastAPI Backend                      │
│  ┌──────────────────────────────────────────┐  │
│  │ Endpoints:                               │  │
│  │  /models  /datasets  /attribute         │  │
│  │  /train-transcoders  /evaluate          │  │
│  └──────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────┐  │
│  │ Job Queue (Celery)                      │  │
│  │  - Attribution tasks                     │  │
│  │  - Transcoder training                   │  │
│  │  - Batch evaluation                      │  │
│  └──────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────┐
│         Circuit-Tracer Core Library             │
│  ┌──────────────┬─────────────┬──────────────┐ │
│  │ Transcoder   │ Attribution │ Intervention │ │
│  │ Training     │ Engine      │ Engine       │ │
│  └──────────────┴─────────────┴──────────────┘ │
└─────────────────────────────────────────────────┘
```

#### Phase 3: Advanced Features (8-12 weeks)

1. **Automated Feature Discovery**
   - Train classifiers to identify task-relevant features
   - Use clustering to find feature families
   - Build a searchable feature database across models

2. **Cross-Model Feature Mapping**
   - Identify equivalent features across different models
   - Track how similar concepts are represented differently
   - Useful for model comparison and transfer learning

3. **Intervention Libraries**
   - Pre-built intervention patterns for common tasks
   - "Steering vectors" for task-specific behavior modification
   - A/B testing framework for model behavior changes

4. **Integration with Existing Tools**
   - Export graphs to Neuronpedia
   - Import SAEs from SAE Lens
   - Connect to model evaluation frameworks (Eleuther Eval, OpenAI Evals)

### Specific Implementation Recommendations

#### 1. Extend `__main__.py` CLI for Batch Processing

```python
# Add new subcommand to circuit_tracer/__main__.py

evaluate_parser = subparsers.add_parser(
    "evaluate",
    help="Batch evaluate model on task dataset"
)
evaluate_parser.add_argument("--model", required=True)
evaluate_parser.add_argument("--transcoder_set", required=True)
evaluate_parser.add_argument("--task", choices=["tool_use", "rag", "reasoning"])
evaluate_parser.add_argument("--dataset", help="Path to custom dataset JSON")
evaluate_parser.add_argument("--output_dir", required=True)
evaluate_parser.add_argument("--metrics", nargs="+", default=["coverage", "coherence"])
```

#### 2. Create Evaluation Module

```python
# circuit_tracer/evaluation/task_evaluator.py

from dataclasses import dataclass
from typing import List, Dict, Any
from circuit_tracer import ReplacementModel, attribute, Graph

@dataclass
class EvaluationResult:
    prompt: str
    graph: Graph
    metrics: Dict[str, float]
    task_features: List[str]
    interpretation: str

class TaskEvaluator:
    """Evaluate how well a model's circuits align with a specific task"""

    def __init__(
        self,
        model: ReplacementModel,
        task_config: Dict[str, Any]
    ):
        self.model = model
        self.task_config = task_config
        self.expected_features = task_config.get("expected_features", [])

    def evaluate_batch(
        self,
        prompts: List[str],
        verbose: bool = True
    ) -> List[EvaluationResult]:
        """Run attribution on batch of prompts and compute metrics"""
        results = []
        for prompt in prompts:
            graph = attribute(prompt, self.model)
            metrics = self._compute_metrics(graph)
            task_features = self._identify_task_features(graph)
            interpretation = self._generate_interpretation(graph, metrics)

            results.append(EvaluationResult(
                prompt=prompt,
                graph=graph,
                metrics=metrics,
                task_features=task_features,
                interpretation=interpretation
            ))
        return results

    def _compute_metrics(self, graph: Graph) -> Dict[str, float]:
        """Compute task-specific evaluation metrics"""
        return {
            "feature_coverage": self._coverage_metric(graph),
            "pathway_coherence": self._coherence_metric(graph),
            "sparsity": self._sparsity_metric(graph),
            "intervention_stability": self._stability_metric(graph)
        }

    # ... implement metric functions
```

#### 3. Create Task Configuration Schema

```yaml
# task_configs/tool_use.yaml

task_name: "tool_use"
description: "Evaluating model circuits for tool/function calling"

expected_features:
  - type: "pattern"
    name: "function_name_extraction"
    description: "Features that activate on function/tool names"

  - type: "pattern"
    name: "parameter_binding"
    description: "Features linking arguments to function parameters"

  - type: "pattern"
    name: "JSON_formatting"
    description: "Features ensuring valid JSON structure"

evaluation_metrics:
  coverage:
    weight: 0.4
    description: "Percentage of expected feature patterns found"

  coherence:
    weight: 0.3
    description: "How well features connect in interpretable pathways"

  stability:
    weight: 0.3
    description: "Robustness to interventions on key features"

intervention_tests:
  - name: "ablate_function_name"
    description: "Zero out function name features, expect output to change"
    expected_change: "high"

  - name: "swap_parameters"
    description: "Modify parameter binding features"
    expected_change: "medium"

datasets:
  - name: "simple_weather_calls"
    path: "data/tool_use/weather.json"
    size: 50

  - name: "complex_multi_tool"
    path: "data/tool_use/multi_tool.json"
    size: 100
```

#### 4. Web API Example (FastAPI)

```python
# api/main.py

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch

from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub
from circuit_tracer.evaluation import TaskEvaluator

app = FastAPI(title="Circuit-Tracer Evaluation API")

# In-memory cache for loaded models (use Redis in production)
model_cache = {}

class AttributionRequest(BaseModel):
    model_name: str
    transcoder_set: str
    prompt: str
    max_n_logits: int = 10

class EvaluationRequest(BaseModel):
    model_name: str
    transcoder_set: str
    task: str
    prompts: List[str]
    metrics: Optional[List[str]] = ["coverage", "coherence"]

@app.post("/api/v1/attribute")
async def run_attribution(request: AttributionRequest):
    """Run attribution on a single prompt"""
    try:
        # Load or get cached model
        cache_key = f"{request.model_name}:{request.transcoder_set}"
        if cache_key not in model_cache:
            transcoder, config = load_transcoder_from_hub(
                request.transcoder_set,
                dtype=torch.float32
            )
            model = ReplacementModel.from_pretrained_and_transcoders(
                request.model_name,
                transcoder,
                dtype=torch.float32
            )
            model_cache[cache_key] = model
        else:
            model = model_cache[cache_key]

        # Run attribution
        graph = attribute(
            prompt=request.prompt,
            model=model,
            max_n_logits=request.max_n_logits
        )

        # Convert to JSON-serializable format
        return {
            "status": "success",
            "input": request.prompt,
            "num_features": len(graph.active_features),
            "graph_data": serialize_graph(graph)  # Implement this
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/evaluate")
async def run_evaluation(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks
):
    """Run batch evaluation on a task dataset"""
    # In production, queue this as a background job
    # For now, return a job ID and process asynchronously
    job_id = create_job_id()

    background_tasks.add_task(
        process_evaluation,
        job_id,
        request
    )

    return {
        "status": "queued",
        "job_id": job_id,
        "message": f"Evaluating {len(request.prompts)} prompts"
    }

@app.get("/api/v1/models")
async def list_models():
    """List available models and their transcoder status"""
    return {
        "models": [
            {
                "name": "gemma-2-2b",
                "transcoders_available": True,
                "transcoder_sets": [
                    "mntss/gemma-scope-transcoders",
                    "mntss/clt-gemma-2-2b-426k"
                ]
            },
            # ... more models
        ]
    }

# ... additional endpoints
```

### Quick Start Guide for Your Use Case

**Week 1-2: Setup & Exploration**
1. Clone and install circuit-tracer
2. Run through existing demos (circuit_tracing_tutorial.ipynb)
3. Test attribution on Gemma-2-2B with tool-use prompts
4. Manually annotate graphs to understand feature patterns

**Week 3-4: First Custom Model**
1. Choose a model you want to evaluate (e.g., Mistral-7B)
2. Train transcoders using EleutherAI Sparsify:
   ```bash
   python -m sparsify mistralai/Mistral-7B-v0.1 --transcode --d_hidden 65536
   ```
3. Export to circuit-tracer format
4. Run attribution on test prompts

**Week 5-6: Build Evaluation Framework**
1. Create task configuration files (tool_use.yaml, rag.yaml)
2. Implement TaskEvaluator class
3. Build metric functions (coverage, coherence, stability)
4. Test on 2-3 models with small prompt sets

**Week 7-10: Basic UI**
1. Set up FastAPI backend with model/dataset endpoints
2. Build React frontend with model selection, dataset upload
3. Integrate existing attribution-graphs-frontend for visualization
4. Add batch processing and results dashboard

**Week 11-12: Iteration & Polish**
1. Test with real evaluation workloads
2. Optimize performance (caching, async processing)
3. Add export formats (CSV, JSON, HTML reports)
4. Document workflows and create user guide

## Key Limitations & Workarounds

### Limitation 1: Requires Pre-Trained Transcoders
**Impact**: Can't analyze arbitrary models immediately
**Workaround**:
- Train transcoders using EleutherAI Sparsify (fastest option)
- Start with supported models (Gemma, Llama, Qwen) to prototype workflows
- Build a transcoder training pipeline early

### Limitation 2: Computational Requirements
**Impact**: Attribution requires GPU (15GB+ for small models, 40GB+ for 7B models)
**Workaround**:
- Use `--offload cpu` or `--offload disk` for memory management
- Batch processing with smaller batch sizes
- Use smaller models (1B-2B) for development, larger for production
- Consider cloud GPU services (Lambda Labs, RunPod, Vast.ai)

### Limitation 3: Transcoder Quality Affects Interpretability
**Impact**: Poorly trained transcoders → less interpretable features
**Workaround**:
- Invest time in hyperparameter tuning for transcoder training
- Validate transcoder reconstruction loss before using for attribution
- Compare multiple transcoder configurations (PLT vs CLT, different feature counts)

### Limitation 4: Manual Feature Interpretation
**Impact**: Understanding what features represent is time-consuming
**Workaround**:
- Leverage Neuronpedia's existing feature descriptions when available
- Build automated feature labeling (activation-based clustering, LLM descriptions)
- Create feature libraries for common concepts across models

### Limitation 5: Task-Specific Evaluation is Custom
**Impact**: No built-in metrics for tool use, RAG, etc.
**Workaround**:
- This is your opportunity! Build the evaluation framework you need
- Start simple: feature presence/absence, graph connectivity
- Iterate based on what correlates with actual task performance

## Additional Resources

### Papers & Research
- [Circuit Tracing: Methods (2025)](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)
- [On the Biology of a Large Language Model (2025)](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
- [Transcoders Beat SAEs for Interpretability (Jan 2025)](https://arxiv.org/abs/2501.18823)
- [Sparse Feature Circuits (2024)](https://www.semanticscholar.org/paper/b8f280d8bf685f8da7c83068e73f000528072d6b)

### Tools & Libraries
- [EleutherAI Sparsify](https://github.com/EleutherAI/sparsify) - Train transcoders/SAEs
- [SAELens](https://github.com/jbloomAus/SAELens) - Comprehensive SAE ecosystem
- [Neuronpedia](https://www.neuronpedia.org/) - Feature visualization platform
- [Attribution Graphs Frontend](https://github.com/anthropics/attribution-graphs-frontend) - React visualization UI

### Community
- TransformerCircuits.pub - Anthropic's interpretability research
- EleutherAI Discord - SAE/transcoder training discussions
- AI Safety Forum - Mechanistic interpretability discussions

## Conclusion

Circuit-tracer is a powerful tool for understanding LLM reasoning, but it requires pre-trained transcoders. To use it for evaluating arbitrary models on tasks like tool use and RAG:

1. **Build a transcoder training pipeline** using EleutherAI Sparsify or SAELens
2. **Create task-specific evaluation frameworks** with metrics for feature coverage, pathway coherence, and intervention stability
3. **Develop a UI** that wraps circuit-tracer with model management, dataset selection, and batch processing
4. **Iterate on metrics** that correlate with actual task performance

The library is well-architected and extensible. Your proposed use case - evaluating model fitness for specific tasks through circuit analysis - is novel and potentially very valuable for model selection and debugging. The main investment is in training transcoders and building evaluation infrastructure around the core attribution engine.

Start with supported models (Gemma-2-2B is fast and accessible), prototype your evaluation metrics, then expand to custom models as you build out the transcoder training pipeline.

---

# Phase 1 Implementation: Evaluation Framework

**Status**: ✅ Complete

Phase 1 has been implemented and includes:
- Task configuration system (YAML-based)
- Evaluation metrics framework
- Task evaluator with batch processing
- CLI integration
- Example task configs (tool_use, rag, reasoning)
- Example datasets
- Transcoder training wrapper

## Quick Start Guide

### 1. Basic Evaluation Example

Evaluate Gemma-2-2B on the tool use task:

```bash
circuit-tracer evaluate \
  --transcoder_set gemma \
  --task_config task_configs/tool_use.yaml \
  --dataset datasets/tool_use/weather.json \
  --output_dir ./evaluation_results \
  --verbose
```

This will:
1. Load the Gemma-2-2B model with GemmaScope transcoders
2. Load the tool_use task configuration
3. Run attribution on each prompt in the weather dataset
4. Compute metrics (coverage, coherence, sparsity, density)
5. Save results to `./evaluation_results/results_tool_use.json`

### 2. Evaluate with Command-Line Prompts

```bash
circuit-tracer evaluate \
  --transcoder_set gemma \
  --task_config task_configs/rag.yaml \
  --prompts \
    "According to the document, what is the main benefit?" \
    "Based on the text, who founded the company?" \
  --output_dir ./eval_results \
  --verbose
```

### 3. Available Task Configurations

**Tool Use** (`task_configs/tool_use.yaml`):
- Expected features: function name extraction, parameter binding, JSON formatting
- Metrics: coverage (30%), coherence (30%), sparsity (20%), density (20%)
- Example datasets: weather calls, multi-tool scenarios

**RAG** (`task_configs/rag.yaml`):
- Expected features: context reference, information extraction, attribution phrases
- Metrics: coverage (35%), coherence (35%), sparsity (15%), density (15%)
- Example datasets: factual QA, multi-doc reasoning

**Reasoning** (`task_configs/reasoning.yaml`):
- Expected features: step markers, logical connectives, intermediate conclusions
- Metrics: coverage (25%), coherence (40%), sparsity (15%), density (20%)
- Example datasets: math word problems, logical puzzles

### 4. Understanding the Output

Results are saved as JSON with this structure:

```json
{
  "task_name": "tool_use",
  "model_name": "google/gemma-2-2b",
  "n_prompts": 5,
  "aggregate_metrics": {
    "coverage_mean": 0.72,
    "coverage_min": 0.65,
    "coverage_max": 0.85,
    "coherence_mean": 0.68,
    "sparsity_mean": 0.91,
    "density_mean": 0.09,
    "weighted_score_mean": 0.70,
    "total_prompts": 5,
    "avg_active_features": 234.5
  },
  "individual_results": [
    {
      "prompt": "What's the weather like in Paris...",
      "metrics": {
        "coverage": 0.75,
        "coherence": 0.68,
        "sparsity": 0.92,
        "density": 0.08,
        "weighted_score": 0.71
      },
      "n_active_features": 245,
      "interpretation": "Strong alignment with task requirements. 245 features active. Pathways show moderate coherence."
    }
  ]
}
```

### 5. Creating Custom Task Configurations

Create a new YAML file in `task_configs/`:

```yaml
task_name: "my_custom_task"
description: "Evaluating model circuits for my specific use case"

expected_features:
  - type: "pattern"
    name: "my_feature"
    description: "What this feature should do"
    matcher:
      activation_threshold: 0.5

evaluation_metrics:
  coverage:
    weight: 0.4
    description: "How well expected features are present"
    enabled: true

  coherence:
    weight: 0.6
    description: "How interpretable the pathways are"
    enabled: true

datasets:
  - name: "my_test_set"
    path: "datasets/my_task/test.json"
    size: 100
```

### 6. Creating Custom Datasets

Datasets are JSON files with this format:

```json
[
  {
    "prompt": "Your test prompt here",
    "expected_answer": "optional answer",
    "expected_features": ["feature1", "feature2"],
    "difficulty": "easy"
  },
  {
    "prompt": "Another test prompt",
    "expected_answer": "another answer",
    "difficulty": "hard"
  }
]
```

The only required field is `"prompt"`. Other fields are metadata for analysis.

### 7. Training Transcoders for New Models

To use circuit-tracer with a model that doesn't have pre-trained transcoders:

**Option A: Using the Python API**

```python
from circuit_tracer.training import TranscoderTrainer, TranscoderTrainingConfig

config = TranscoderTrainingConfig(
    model_name="mistralai/Mistral-7B-v0.1",
    dataset="c4",
    d_transcoder=65536,  # 16x expansion
    n_tokens=100_000_000,
    output_dir="./my_transcoders",
    use_transcoders=True,  # True for transcoders, False for SAEs
)

trainer = TranscoderTrainer(config)
output_dir = trainer.train(method="sparsify", verbose=True)

# Export for circuit-tracer
from circuit_tracer.training import export_for_circuit_tracer

export_for_circuit_tracer(
    transcoder_dir=output_dir,
    model_name="mistralai/Mistral-7B-v0.1",
    output_dir="./mistral_transcoders_ct_format",
)
```

**Option B: Using EleutherAI Sparsify directly**

```bash
# Install sparsify
pip install eleutherai-sparsify

# Train transcoders
python -m sparsify mistralai/Mistral-7B-v0.1 c4 \
  --transcode \
  --d_hidden 65536 \
  --n_tokens 100000000 \
  --save_dir ./my_transcoders

# Then export to circuit-tracer format
python -c "
from circuit_tracer.training import export_for_circuit_tracer
export_for_circuit_tracer(
    './my_transcoders',
    'mistralai/Mistral-7B-v0.1',
    './mistral_ct_format'
)
"
```

**Then use your custom transcoders:**

```bash
circuit-tracer evaluate \
  --model mistralai/Mistral-7B-v0.1 \
  --transcoder_set ./mistral_ct_format \
  --task_config task_configs/tool_use.yaml \
  --dataset datasets/tool_use/weather.json \
  --output_dir ./results
```

### 8. Programmatic Usage (Python API)

```python
from circuit_tracer import ReplacementModel
from circuit_tracer.evaluation import TaskEvaluator, load_task_config
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub
import torch

# Load model and transcoders
transcoder, config = load_transcoder_from_hub("gemma", dtype=torch.float32)
model = ReplacementModel.from_pretrained_and_transcoders(
    "google/gemma-2-2b",
    transcoder,
    dtype=torch.float32
)

# Load task configuration
task_config = load_task_config("task_configs/tool_use.yaml")

# Create evaluator
evaluator = TaskEvaluator(
    model=model,
    task_config=task_config,
    attribution_kwargs={
        "max_n_logits": 10,
        "batch_size": 256,
    }
)

# Evaluate single prompt
result = evaluator.evaluate_prompt(
    "What's the weather in Paris? Use get_weather function."
)

print(f"Metrics: {result.metrics}")
print(f"Interpretation: {result.interpretation}")
print(f"Active features: {len(result.graph.active_features)}")

# Evaluate batch
prompts = [
    "Weather in Tokyo?",
    "Show me forecast for London",
    "Temperature in NYC tomorrow?"
]

batch_results = evaluator.evaluate_batch(prompts, verbose=True)
print(f"Aggregate metrics: {batch_results.aggregate_metrics}")

# Save results
import json
with open("results.json", "w") as f:
    json.dump(batch_results.summary(), f, indent=2)
```

### 9. Customizing Evaluation Metrics

You can create custom metrics by extending the metrics module:

```python
# In your own code
from circuit_tracer.graph import Graph
from circuit_tracer.evaluation.task_evaluator import TaskEvaluator

def my_custom_metric(graph: Graph) -> float:
    """Compute a custom metric based on graph properties."""
    # Example: measure average feature activation strength
    activations = graph.activation_values
    return float(activations.mean())

# Then use it with TaskEvaluator
class CustomTaskEvaluator(TaskEvaluator):
    def _compute_metrics(self, graph: Graph):
        metrics = super()._compute_metrics(graph)
        metrics["custom_metric"] = my_custom_metric(graph)
        return metrics
```

### 10. Comparing Multiple Models

```bash
# Evaluate Gemma-2-2B
circuit-tracer evaluate \
  --transcoder_set gemma \
  --task_config task_configs/reasoning.yaml \
  --dataset datasets/reasoning/math_word_problems.json \
  --output_dir ./results/gemma \
  --verbose

# Evaluate Llama-3.2-1B
circuit-tracer evaluate \
  --transcoder_set llama \
  --model meta-llama/Llama-3.2-1B \
  --task_config task_configs/reasoning.yaml \
  --dataset datasets/reasoning/math_word_problems.json \
  --output_dir ./results/llama \
  --verbose

# Compare results
python -c "
import json

with open('./results/gemma/results_reasoning.json') as f:
    gemma = json.load(f)
with open('./results/llama/results_reasoning.json') as f:
    llama = json.load(f)

print('Gemma-2-2B weighted_score:', gemma['aggregate_metrics']['weighted_score_mean'])
print('Llama-3.2-1B weighted_score:', llama['aggregate_metrics']['weighted_score_mean'])
"
```

## Next Steps for Phase 2

With Phase 1 complete, you can now:

1. **Test the evaluation framework** on supported models (Gemma, Llama, Qwen)
2. **Create task configs** for your specific use cases
3. **Build datasets** for tool use, RAG, or other tasks you want to evaluate
4. **Train transcoders** for custom models using the training wrapper
5. **Begin Phase 2**: Build the web UI for easier model/dataset management

The evaluation infrastructure is fully functional and ready to use!
