# Fine-Tuning Evaluation Guide

This guide explains how to use circuit-tracer to monitor and evaluate fine-tuning by comparing checkpoints against the base model using base model transcoders.

## Can You Use Base Model Transcoders on Fine-Tuned Models?

**Yes!** With important caveats:

### When It Works Well:
- **Light fine-tuning** (LoRA, few epochs, small learning rates)
- **Task-specific fine-tuning** that builds on existing features
- **Early-to-mid** training checkpoints

### When It Degrades:
- Heavy parameter updates over many epochs
- Significant distribution shift from base model training data
- Catastrophic forgetting of base model capabilities

## Key Use Cases

### 1. Monitor Fine-Tuning Progress

Track which features activate during training to verify the model is learning expected patterns:

```bash
circuit-tracer compare-checkpoints \
  --base_model google/gemma-2-2b \
  --transcoder_set gemma \
  --checkpoints ./training/checkpoint-100 ./training/checkpoint-200 ./training/checkpoint-300 \
  --task_config task_configs/tool_use.yaml \
  --dataset datasets/tool_use/weather.json \
  --output_dir ./finetuning_analysis
```

### 2. Detect Catastrophic Forgetting

If important base features disappear, you're losing capabilities:

```python
from circuit_tracer.evaluation import FineTuneComparator, load_task_config

comparator = FineTuneComparator(
    base_model="google/gemma-2-2b",
    transcoder_set="gemma",
    task_config=load_task_config("task_configs/tool_use.yaml")
)

analysis = comparator.compare_checkpoints(
    checkpoint_paths=["./ckpt-100", "./ckpt-200"],
    eval_prompts=my_prompts
)

# Check for feature loss
for ckpt in analysis.checkpoints:
    if ckpt.drift_metrics.lost_features_count > 50:
        print(f"⚠️ {ckpt.checkpoint_name}: Lost {ckpt.drift_metrics.lost_features_count} base features!")
```

### 3. Decide When to Stop Training

Monitor transcoder validity and circuit coherence to find the optimal checkpoint:

```python
# Track validity over training
validities = [ckpt.drift_metrics.transcoder_validity for ckpt in analysis.checkpoints]

# Find best checkpoint
best_idx = max(range(len(analysis.checkpoints)),
               key=lambda i: analysis.checkpoints[i].metric_deltas.get("coherence", 0))

print(f"Best checkpoint: {analysis.checkpoints[best_idx].checkpoint_name}")
```

## Understanding the Metrics

### Transcoder Validity (0.0 - 1.0)
- **> 0.8**: Transcoders still highly valid, features interpretable
- **0.6 - 0.8**: Moderate drift, most features still meaningful
- **< 0.6**: Significant drift, consider retraining transcoders

### Feature Overlap
Percentage of base model features still active after fine-tuning:
- High overlap: Building on base model knowledge
- Low overlap: Significant divergence (could be good or bad depending on goal)

### Activation Correlation
How similarly features activate compared to base model:
- High correlation: Similar internal representations
- Low correlation: Different activation patterns (new behaviors)

### New Features Count
Features unique to the fine-tuned model:
- Expected for task-specific learning
- Track to see if model is learning new patterns

### Lost Features Count
Base features no longer active:
- Monitor for catastrophic forgetting
- Some loss is normal, excessive loss is concerning

## Example Workflow

### During Fine-Tuning

```python
from transformers import Trainer, TrainerCallback
from circuit_tracer.evaluation import FineTuneComparator, load_task_config

task_config = load_task_config("task_configs/tool_use.yaml")
comparator = FineTuneComparator(
    base_model="google/gemma-2-2b",
    transcoder_set="gemma",
    task_config=task_config
)

class CircuitMonitorCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = f"{args.output_dir}/checkpoint-{state.global_step}"

        # Quick evaluation
        result = comparator.evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            eval_prompts=["What's the weather in Paris?"],
            checkpoint_name=f"step-{state.global_step}"
        )

        # Log to wandb/tensorboard
        wandb.log({
            "circuit/coverage": result["metrics"]["coverage_mean"],
            "circuit/coherence": result["metrics"]["coherence_mean"],
        }, step=state.global_step)

trainer = Trainer(
    model=model,
    callbacks=[CircuitMonitorCallback()],
    ...
)
```

### After Fine-Tuning

```bash
# Compare all checkpoints
circuit-tracer compare-checkpoints \
  --base_model google/gemma-2-2b \
  --transcoder_set gemma \
  --checkpoints ./training/checkpoint-* \
  --task_config task_configs/tool_use.yaml \
  --dataset datasets/tool_use/weather.json \
  --output_dir ./final_analysis

# Discover available SAEs for your model
circuit-tracer discover-saes --model llama

# Use different transcoders
circuit-tracer compare-checkpoints \
  --base_model meta-llama/Llama-3.2-1B \
  --transcoder_set llama \
  --checkpoints ./llama_finetuned/checkpoint-500 \
  --task_config task_configs/reasoning.yaml \
  --eval_prompts "Solve: 2x + 5 = 13" \
  --output_dir ./llama_analysis
```

## Interpreting Results

### Example Output:

```
==================================================
CHECKPOINT COMPARISON SUMMARY
==================================================
Task: tool_use
Checkpoints analyzed: 3

Best coverage: checkpoint-300
Best coherence: checkpoint-200
Final transcoder validity: 0.76
Validity trend: degrading
==================================================

Detailed results:

1. checkpoint-100
   ✅ Transcoders remain highly valid. Learned 23 new features. Improved: coverage.
   Metric changes:
     coverage: +0.085
     coherence: +0.012
     sparsity: -0.003

2. checkpoint-200
   ✅ Transcoders remain highly valid. Learned 45 new features. Improved: coverage, coherence.
   Metric changes:
     coverage: +0.142
     coherence: +0.089
     sparsity: -0.012

3. checkpoint-300
   ⚠️ Transcoders moderately degraded. ⚠️ Lost 18 base features (possible forgetting). Improved: coverage.
   Metric changes:
     coverage: +0.156
     coherence: +0.034
     sparsity: -0.028
```

### What This Tells You:
- **Checkpoint 200** is likely optimal: high coherence, transcoders still valid
- **Checkpoint 300** shows signs of overfitting/forgetting
- Model is successfully learning task features (coverage improving)
- Some base features being lost at checkpoint 300 (investigate further)

## When to Retrain Transcoders

Consider retraining transcoders on your fine-tuned model if:

1. **Transcoder validity < 0.6** across multiple prompts
2. **Heavy fine-tuning** (many epochs, full parameter updates)
3. **You plan to fine-tune further** and need accurate monitoring

Quick transcoder retraining:
```python
from circuit_tracer.training import TranscoderTrainer, TranscoderTrainingConfig

config = TranscoderTrainingConfig(
    model_name="./my_finetuned_model",
    d_transcoder=65536,
    n_tokens=10_000_000,  # Less than full training
    output_dir="./finetuned_transcoders"
)

trainer = TranscoderTrainer(config)
trainer.train(method="sparsify")
```

## Advanced: Feature-Guided Fine-Tuning

Use circuit insights to guide training (experimental):

```python
# Identify critical features for your task
from circuit_tracer.evaluation import TaskFeatureDiscovery

discoverer = TaskFeatureDiscovery(model, task_config)
critical_features = discoverer.discover_critical_features(
    task_prompts=tool_use_examples,
    baseline_prompts=general_examples
)

# During training, monitor if these features are activating
# Add custom loss to preserve important base features
# Regularize to encourage task-specific feature activation
```

## Available Models with SAEs

Run `circuit-tracer discover-saes` to see all available models:

```
Currently Supported (Native):
- Gemma 2 (2B, 9B)
- Llama 3.2 (1B)
- Llama 3 / 3.1 (8B) - via SAELens
- Qwen 3 (0.6B - 14B)
- Mistral 7B - via SAELens
- GPT-2 - via SAELens
- Phi-3 Mini - via SAELens
```

## Best Practices

1. **Evaluate multiple prompts** - Don't rely on a single prompt
2. **Track trends** - One checkpoint is noisy, look at the trajectory
3. **Compare to base early** - Establish baseline before training
4. **Save checkpoints frequently** - Especially early in training
5. **Use task-appropriate metrics** - Coverage matters for skill acquisition, coherence for robustness
6. **Monitor both gains and losses** - New features AND preserved base features

## FAQ

**Q: My transcoder validity is 0.5, should I stop?**
A: Not necessarily. If your task metrics are improving and you're achieving your goal, the base transcoders may just not capture your fine-tuned behavior well. Consider this as a signal to retrain transcoders if you want continued interpretability.

**Q: Can I compare LoRA adapters this way?**
A: Yes! LoRA typically preserves base model features well, so transcoder validity should remain high.

**Q: What if I fine-tuned on a completely different domain?**
A: Expect lower transcoder validity and feature overlap. This is normal for domain transfer. You might want to train new transcoders for the fine-tuned model.

**Q: How expensive is checkpoint comparison?**
A: Similar cost to regular evaluation. For Gemma-2-2B, expect ~5-10 minutes per checkpoint on a single GPU.

---

For more details on the evaluation framework, see `CODEBASE_SUMMARY.md`.
