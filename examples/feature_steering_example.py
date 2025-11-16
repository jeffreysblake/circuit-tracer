"""Example: Feature Steering with Circuit-Tracer.

This example demonstrates how to use circuit-tracer's feature steering
capabilities to control model behavior at inference time.
"""

import torch

from circuit_tracer import ReplacementModel
from circuit_tracer.steering import FeatureSteering
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub


def example_1_semantic_discovery():
    """Example 1: Discover features by semantic description."""
    print("=" * 70)
    print("EXAMPLE 1: Semantic Feature Discovery")
    print("=" * 70)

    # Load model
    transcoder, config = load_transcoder_from_hub(
        "mntss/gemma-scope-transcoders", dtype=torch.float32
    )
    model = ReplacementModel.from_pretrained_and_transcoders(
        "google/gemma-2-2b", transcoder, dtype=torch.float32
    )

    # Initialize steering
    steering = FeatureSteering(model)

    # Discover "concise" features
    print("\n🔍 Discovering 'concise and direct' features...")
    features = steering.discover_features(
        description="concise and direct responses without hedging",
        n_features=10,
        reference_prompts=[
            "Yes, the answer is 42.",
            "Quantum computers use superposition.",
            "Paris is the capital of France.",
        ],
    )

    print(f"\n✅ Discovered {len(features)} features:")
    for i, feat in enumerate(features, 1):
        layer, pos, feat_idx = feat
        print(f"  {i}. Layer {layer}, Position {pos}, Feature {feat_idx}")


def example_2_contrastive_discovery():
    """Example 2: Discover features using contrastive examples."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Contrastive Feature Discovery")
    print("=" * 70)

    # Load model
    transcoder, config = load_transcoder_from_hub(
        "mntss/gemma-scope-transcoders", dtype=torch.float32
    )
    model = ReplacementModel.from_pretrained_and_transcoders(
        "google/gemma-2-2b", transcoder, dtype=torch.float32
    )

    steering = FeatureSteering(model)

    # Define desired vs undesired behavior
    desired = [
        "The answer is 42.",
        "Quantum computers use superposition.",
        "This is correct.",
    ]

    undesired = [
        "Well, it might possibly be around 42, but I'm not entirely sure.",
        "Quantum computers could potentially use something like superposition.",
        "This seems like it could be correct, maybe.",
    ]

    print("\n🔍 Finding features that distinguish concise from hedging language...")
    features = steering.discover_features_contrastive(
        desired_examples=desired,
        undesired_examples=undesired,
        top_k=15,
    )

    print(f"\n✅ Discovered {len(features)} differentiating features:")
    for i, (feat, weight) in enumerate(features[:5], 1):  # Show top 5
        layer, pos, feat_idx = feat
        print(f"  {i}. Layer {layer}, Position {pos}, Feature {feat_idx}")
        print(f"     Suggested weight: {weight:+.3f}")

    # Apply steering
    print("\n⚙️  Applying steering...")
    for feat, weight in features:
        steering.set_feature_weight(feat, weight)

    # Test
    test_prompt = "What is quantum computing?"
    print(f"\n📊 Testing on: '{test_prompt}'")

    original, steered = steering.analyze_with_steering(test_prompt, verbose=True)


def example_3_auto_steer():
    """Example 3: Automatic feature steering."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Auto-Steer")
    print("=" * 70)

    # Load model
    transcoder, config = load_transcoder_from_hub(
        "mntss/gemma-scope-transcoders", dtype=torch.float32
    )
    model = ReplacementModel.from_pretrained_and_transcoders(
        "google/gemma-2-2b", transcoder, dtype=torch.float32
    )

    steering = FeatureSteering(model)

    # Auto-steer: Just describe what you want
    print("\n🤖 Auto-steering: 'be more humorous and playful'")

    edits = steering.auto_steer(
        specification="be more humorous and playful",
        reference_prompts=[
            "Why did the chicken cross the road?",
            "Let's have some fun with this!",
        ],
        n_features=10,
        base_weight=0.3,
    )

    print(f"\n✅ Configured {len(edits)} features automatically")

    # Visualize
    viz = steering.visualize_steering()
    print("\n" + viz)


def example_4_conditional_steering():
    """Example 4: Conditional steering based on context."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Conditional Steering")
    print("=" * 70)

    # Load model
    transcoder, config = load_transcoder_from_hub(
        "mntss/gemma-scope-transcoders", dtype=torch.float32
    )
    model = ReplacementModel.from_pretrained_and_transcoders(
        "google/gemma-2-2b", transcoder, dtype=torch.float32
    )

    steering = FeatureSteering(model)

    # Define condition
    def is_technical_query(text: str) -> bool:
        technical_keywords = ["quantum", "algorithm", "neural", "cryptography"]
        return any(kw in text.lower() for kw in technical_keywords)

    # Discover "simple explanation" features
    simple_features = steering.discover_features(
        description="explain simply for beginners",
        n_features=5,
    )

    # Set up conditional steering
    print("\n⚙️  Setting up conditional steering...")
    print("   Trigger: Technical keywords detected")
    print("   Action: Amplify 'simple explanation' features")

    steering.set_conditional_steering(
        condition=is_technical_query,
        features={feat: 0.5 for feat in simple_features},
        description="Simplify technical explanations",
    )

    # Test prompts
    test_prompts = [
        "What is quantum computing?",  # Should trigger
        "What's your favorite color?",  # Should not trigger
    ]

    for prompt in test_prompts:
        print(f"\n📝 Prompt: '{prompt}'")
        triggered = is_technical_query(prompt)
        print(f"   Conditional steering: {'✅ ACTIVE' if triggered else '❌ INACTIVE'}")


def example_5_compare_behaviors():
    """Example 5: Compare different steering configurations."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Compare Steering Configurations")
    print("=" * 70)

    # Load model
    transcoder, config = load_transcoder_from_hub(
        "mntss/gemma-scope-transcoders", dtype=torch.float32
    )
    model = ReplacementModel.from_pretrained_and_transcoders(
        "google/gemma-2-2b", transcoder, dtype=torch.float32
    )

    steering = FeatureSteering(model)

    # Define different configurations
    configs = {
        "concise": {(15, 10, 1234): 0.5, (15, 10, 5678): 0.3},
        "verbose": {(15, 10, 1234): -0.5, (15, 10, 9012): 0.4},
        "technical": {(18, 5, 2345): 0.6, (18, 5, 6789): 0.5},
    }

    # Test prompt
    test_prompt = "Explain how neural networks work"

    print(f"\n📊 Comparing configurations on: '{test_prompt}'")

    results = steering.compare_behaviors(
        prompt=test_prompt,
        steering_configs=configs,
        verbose=True,
    )


def example_6_inspect_activations():
    """Example 6: Inspect feature activations."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Inspect Feature Activations")
    print("=" * 70)

    # Load model
    transcoder, config = load_transcoder_from_hub(
        "mntss/gemma-scope-transcoders", dtype=torch.float32
    )
    model = ReplacementModel.from_pretrained_and_transcoders(
        "google/gemma-2-2b", transcoder, dtype=torch.float32
    )

    steering = FeatureSteering(model)

    # Set some steering
    steering.set_feature_weight((15, 10, 1234), 0.5)
    steering.set_feature_weight((15, 10, 5678), -0.3)

    # Inspect
    prompt = "What is machine learning?"
    print(f"\n🔍 Inspecting activations for: '{prompt}'")

    info = steering.inspect_activations(prompt, top_k=10)

    print(f"\n📊 Results:")
    print(f"   Total active features: {info['total_active']}")
    print(f"   Total activation: {info['total_activation']:.3f}")

    print(f"\n   Top 5 features:")
    for feat, activation in info["top_features"][:5]:
        layer, pos, feat_idx = feat
        print(f"     L{layer} P{pos} F{feat_idx}: {activation:.3f}")

    if info["steered_features"]:
        print(f"\n   Steered features:")
        for item in info["steered_features"]:
            feat = item["feature"]
            layer, pos, feat_idx = feat
            print(f"     L{layer} P{pos} F{feat_idx}:")
            print(f"       Original: {item['activation']:.3f}")
            print(f"       Steered: {item['steered_activation']:.3f}")
            print(f"       Weight: {item['steering_weight']:+.2f}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CIRCUIT-TRACER FEATURE STEERING EXAMPLES")
    print("=" * 70)

    # Run examples
    example_1_semantic_discovery()
    example_2_contrastive_discovery()
    example_3_auto_steer()
    example_4_conditional_steering()
    example_5_compare_behaviors()
    example_6_inspect_activations()

    print("\n" + "=" * 70)
    print("✅ All examples completed!")
    print("=" * 70)
