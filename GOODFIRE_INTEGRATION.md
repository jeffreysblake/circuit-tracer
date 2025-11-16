# Goodfire Integration Guide

This document explains Goodfire's Feature Steering approach and how to achieve similar functionality with circuit-tracer.

## What is Goodfire?

**Goodfire Ember** is a hosted API/SDK (released Dec 2024) that enables direct control over AI model behavior by manipulating internal computational units called "features." It uses Sparse Autoencoders (SAEs) to decompose model activations into interpretable components.

### Key Capabilities:
- **Auto Steer**: Describe desired behavior ("be funny"), automatically find and adjust features
- **Feature Search**: Semantic search across available features
- **Contrastive Search**: Compare datasets to identify differentiating features
- **Feature Modification**: Amplify (positive weight) or suppress (negative weight) specific features
- **Conditional Steering**: Apply feature modifications based on context
- **Feature Inspection**: Analyze which features activate during generation

### Supported Models:
- Meta-Llama-3.1-8B-Instruct
- Meta-Llama-3.3-70B-Instruct

### Open-Source SAEs:
Goodfire released state-of-the-art SAEs in January 2025:
- `Goodfire/Llama-3.1-8B-Instruct-SAE-l19` (Layer 19, L0=91)
- `Goodfire/Llama-3.3-70B-Instruct-SAE-l50` (Layer 50, L0=121)

Trained on LMSYS-Chat-1M dataset with toxic features removed.

---

## Goodfire vs Circuit-Tracer

| Feature | Goodfire Ember | Circuit-Tracer |
|---------|----------------|----------------|
| **Deployment** | Hosted API (requires API key) | Local, open-source |
| **Cost** | Paid service | Free |
| **Models** | Llama 3.1 8B, 3.3 70B | Any model (requires transcoders) |
| **SAE Support** | Per-layer SAEs | Per-layer + cross-layer transcoders |
| **Feature Steering** | Built-in API | Implement yourself (see below) |
| **Circuit Analysis** | Limited | Full attribution graphs |
| **Privacy** | Cloud-based | Fully local |
| **Ease of Use** | Very easy (hosted) | Requires setup |

**Key Difference**: Goodfire is optimized for **runtime steering** (modify behavior during generation), while circuit-tracer excels at **analysis and understanding** (why did the model generate this?).

---

## Integration Options

### Option 1: Use Goodfire's Hosted API (Easiest)

If you want Goodfire's steering capabilities with minimal setup:

```bash
pip install goodfire
```

```python
import goodfire

# Initialize client
client = goodfire.Client(api_key="YOUR_API_KEY")
variant = goodfire.Variant("meta-llama/Llama-3.1-8B-Instruct")

# Auto-steer: Describe what you want
edits = client.features.AutoSteer(
    specification="respond concisely without hedging",
    model=variant,
)
variant.set(edits)

# Generate with modified behavior
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "What is quantum computing?"}],
    model=variant,
)
```

**Pros**:
- No setup required
- State-of-the-art SAEs
- Easy auto-steering

**Cons**:
- Requires API key and payment
- Cloud-based (privacy concerns)
- Limited to Llama models

---

### Option 2: Use Goodfire's Open-Source SAEs Locally

Goodfire released their SAEs on HuggingFace. You can potentially use them with circuit-tracer:

```python
from transformers import AutoModel
import torch

# Load Goodfire's SAE
sae = AutoModel.from_pretrained("Goodfire/Llama-3.1-8B-Instruct-SAE-l19")

# TODO: Convert to circuit-tracer transcoder format
# This requires understanding their SAE architecture
```

**Status**: ⚠️ **Not yet implemented** - Goodfire's SAE format may differ from circuit-tracer's transcoder format. Format conversion would be needed.

**Pros**:
- Free, local usage
- State-of-the-art SAEs

**Cons**:
- Format conversion required
- Only for specific Llama models
- Limited to single layers (L19, L50)

---

### Option 3: Implement Feature Steering with Circuit-Tracer (Recommended)

Circuit-tracer can achieve similar steering functionality using its existing transcoders. We've implemented a **FeatureSteering** module that provides:

- **Feature discovery** via attribution analysis
- **Runtime feature modification** during generation
- **Conditional steering** based on context
- **Feature inspection** to understand activations

See implementation below.

**Pros**:
- Works with ANY model (if you have transcoders)
- Fully local and private
- Free and open-source
- Combines steering with circuit analysis

**Cons**:
- Requires training transcoders (if not available)
- More setup than hosted API

---

## Circuit-Tracer Feature Steering Implementation

We've implemented a local feature steering module inspired by Goodfire's approach:

### Installation

```bash
# Already included in circuit-tracer
```

### Quick Start

```python
from circuit_tracer import ReplacementModel
from circuit_tracer.steering import FeatureSteering
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

# Load model with transcoders
transcoder, config = load_transcoder_from_hub("mntss/gemma-scope-transcoders")
model = ReplacementModel.from_pretrained_and_transcoders(
    "google/gemma-2-2b",
    transcoder,
)

# Initialize steering
steering = FeatureSteering(model)

# Discover features related to a concept
features = steering.discover_features(
    description="technical jargon and complex terminology",
    n_features=10,
)

# Apply steering: suppress jargon features
steering.set_feature_weight(features[0], weight=-0.5)  # Suppress
steering.set_feature_weight(features[1], weight=-0.3)

# Generate with modified behavior
output = model.generate("Explain quantum computing", max_length=100)
print(output)

# Inspect what happened
activations = steering.inspect_activations(output)
print(f"Active features: {len(activations)}")
```

### Advanced: Conditional Steering

```python
# Apply steering only when certain conditions are met
def is_technical_query(text):
    keywords = ["quantum", "algorithm", "neural", "optimization"]
    return any(kw in text.lower() for kw in keywords)

# Set conditional rule
steering.set_conditional_steering(
    condition=is_technical_query,
    features={
        (15, 10, 1234): 0.5,  # Amplify "explain simply" feature
        (15, 10, 5678): -0.4, # Suppress "technical jargon" feature
    }
)

# This will apply steering automatically when condition matches
```

### Feature Discovery Methods

Circuit-tracer provides three discovery methods similar to Goodfire:

#### 1. Semantic Discovery (like Goodfire's Feature Search)

```python
# Find features by semantic description
features = steering.discover_features(
    description="humorous and playful tone",
    n_features=15,
    method="semantic",
)
```

#### 2. Contrastive Discovery (like Goodfire's Contrastive Search)

```python
# Find features that differentiate two datasets
desired_prompts = [
    "Explain like I'm five: What is gravity?",
    "Simple explanation: How do planes fly?",
]

undesired_prompts = [
    "The gravitational force is described by Einstein's field equations...",
    "Aerodynamic lift results from Bernoulli's principle and pressure differentials...",
]

features = steering.discover_features_contrastive(
    desired_examples=desired_prompts,
    undesired_examples=undesired_prompts,
    top_k=20,
)

# Apply discovered features
for feat, weight in features:
    steering.set_feature_weight(feat, weight)
```

#### 3. Task-Specific Discovery (Circuit-Tracer Exclusive)

```python
# Use TaskFeatureDiscovery for automatic identification
from circuit_tracer.rag import TaskFeatureDiscovery

discovery = TaskFeatureDiscovery(model)
result = discovery.discover_critical_features(
    task_prompts=["Call get_weather for New York", "Call send_email to john@example.com"],
    baseline_prompts=["What's the weather?", "Send an email"],
    top_k=15,
)

# Apply task-critical features
for feat in result["features"]:
    steering.set_feature_weight(feat, weight=0.3)
```

---

## Use Cases

### 1. Reduce Hedging in Responses

```python
# Discover "hedging language" features
hedging_features = steering.discover_features_contrastive(
    desired_examples=["Yes, quantum computers use superposition."],
    undesired_examples=["Quantum computers might potentially use superposition in some cases."],
)

# Suppress hedging
for feat, weight in hedging_features:
    steering.set_feature_weight(feat, -0.4)
```

### 2. Enforce JSON Output Format

```python
# Find "JSON formatting" features
json_features = steering.discover_features(
    description="structured JSON output with proper syntax",
    n_features=10,
)

# Amplify JSON features
for feat in json_features:
    steering.set_feature_weight(feat, 0.6)
```

### 3. Conditional Safety Steering

```python
def is_jailbreak_attempt(text):
    jailbreak_patterns = ["ignore previous", "disregard", "pretend you are"]
    return any(pattern in text.lower() for pattern in jailbreak_patterns)

# Amplify refusal features when jailbreak detected
steering.set_conditional_steering(
    condition=is_jailbreak_attempt,
    features={(12, 5, 891): 1.5},  # "Refusal" feature
)
```

---

## Comparison to Other Tools

### Anthropic's Feature Steering

Anthropic published research on feature steering for Claude (Dec 2024), focusing on:
- Safety features (prevent harmful outputs)
- Behavioral traits (sycophancy, deception)
- Subject-matter features (specific knowledge domains)

Circuit-tracer can replicate this by:
1. Discovering safety/behavioral features via contrastive search
2. Applying steering at inference time
3. Monitoring feature drift during fine-tuning

### SAELens

SAELens provides tools for training SAEs but limited steering capabilities. Circuit-tracer integrates with SAELens-trained SAEs and adds:
- Circuit analysis and visualization
- Cross-layer feature tracking
- Task-specific evaluation
- Automated feature discovery

---

## Training Custom Transcoders for Steering

If you want to use feature steering on models not supported by Goodfire:

```bash
# Train transcoders for your model
circuit-tracer train \
  --model facebook/opt-1.3b \
  --output_dir ./my-opt-transcoders \
  --n_tokens 100000000 \
  --d_transcoder 65536

# Use for steering
from circuit_tracer.steering import FeatureSteering

model = ReplacementModel.from_pretrained_and_transcoders(
    "facebook/opt-1.3b",
    "./my-opt-transcoders",
)

steering = FeatureSteering(model)
# ... use as shown above
```

---

## Limitations

### Circuit-Tracer Steering Limitations:
- Requires trained transcoders (training takes time and compute)
- Feature discovery is slower than Goodfire's hosted API
- No pre-trained "concept bank" like Goodfire

### Goodfire Limitations:
- Only supports Llama 3.1 8B and 3.3 70B
- Hosted API requires payment
- Limited circuit analysis (focused on steering, not understanding)
- Cloud-based (privacy concerns for sensitive data)

---

## Recommendations

**Choose Goodfire if:**
- You're working with Llama 3.1/3.3 models
- You need immediate steering without setup
- You're okay with cloud-based processing
- You value ease of use over customization

**Choose Circuit-Tracer if:**
- You need to work with diverse models
- You want full circuit analysis and understanding
- You require local/private deployment
- You're doing research and need flexibility
- You want to monitor fine-tuning

**Best of Both Worlds:**
Use circuit-tracer for analysis and understanding, then implement steering based on insights. For production Llama deployments, consider Goodfire's hosted API.

---

## Future Work

1. **Format Converter**: Build a tool to convert Goodfire's open-source SAEs to circuit-tracer transcoder format
2. **Concept Bank**: Create a library of pre-discovered features for common behaviors
3. **Hybrid Approach**: Use Goodfire for Llama models, circuit-tracer for others
4. **Automated Steering**: Auto-discover and apply steering based on task objectives

---

## Resources

- **Goodfire Documentation**: https://docs.goodfire.ai
- **Goodfire SDK**: https://github.com/goodfire-ai/goodfire-sdk
- **Open-Source SAEs**: https://huggingface.co/Goodfire
- **Circuit-Tracer Steering**: `circuit_tracer/steering/` (this implementation)
- **Research**: Goodfire's blog at https://www.goodfire.ai/blog

---

## Citation

If you use Goodfire's open-source SAEs:

```bibtex
@misc{goodfire2025saes,
  title={Open-Source Sparse Autoencoders for Llama 3.1 8B and Llama 3.3 70B},
  author={Goodfire AI},
  year={2025},
  url={https://www.goodfire.ai/blog/sae-open-source-announcement}
}
```

If you use circuit-tracer's steering implementation:

```bibtex
@software{circuit_tracer_steering,
  title={Circuit-Tracer Feature Steering},
  author={Circuit-Tracer Contributors},
  year={2025},
  url={https://github.com/your-repo/circuit-tracer}
}
```
