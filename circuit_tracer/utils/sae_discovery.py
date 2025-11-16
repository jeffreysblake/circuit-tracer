"""Discovery and integration with SAELens pretrained SAEs registry."""

import logging
from dataclasses import dataclass
from typing import Any

import requests
import yaml

logger = logging.getLogger(__name__)

SAELENS_REGISTRY_URL = (
    "https://raw.githubusercontent.com/jbloomAus/SAELens/main/sae_lens/pretrained_saes.yaml"
)


@dataclass
class SAEInfo:
    """Information about an available SAE."""

    name: str
    model: str
    repo_id: str
    layers: list[int] | None
    hook_point_type: str  # e.g., "resid_post", "mlp_out", "attn_out"
    d_sae: int | None  # Number of features
    expansion_factor: int | None
    description: str = ""
    source: str = "saelens"  # "saelens", "circuit-tracer", "custom"


class SAERegistry:
    """Discover and manage available SAEs from multiple sources."""

    def __init__(self):
        """Initialize the SAE registry."""
        self.saelens_saes: dict[str, Any] = {}
        self.circuit_tracer_saes: dict[str, SAEInfo] = {}
        self._load_saelens_registry()
        self._load_circuit_tracer_saes()

    def _load_saelens_registry(self):
        """Load the SAELens pretrained SAEs registry."""
        try:
            response = requests.get(SAELENS_REGISTRY_URL, timeout=10)
            response.raise_for_status()
            self.saelens_saes = yaml.safe_load(response.text) or {}
            logger.info(f"Loaded {len(self.saelens_saes)} SAE entries from SAELens registry")
        except Exception as e:
            logger.warning(f"Failed to load SAELens registry: {e}")
            self.saelens_saes = {}

    def _load_circuit_tracer_saes(self):
        """Load circuit-tracer native transcoders."""
        # These are the ones already supported by circuit-tracer
        native_transcoders = [
            SAEInfo(
                name="gemma-2-2b-gemma-scope",
                model="google/gemma-2-2b",
                repo_id="mntss/gemma-scope-transcoders",
                layers=list(range(26)),  # Gemma 2 2B has 26 layers
                hook_point_type="mlp",
                d_sae=None,  # Variable per layer
                expansion_factor=None,
                description="GemmaScope PLT transcoders for Gemma 2 2B",
                source="circuit-tracer",
            ),
            SAEInfo(
                name="gemma-2-2b-clt-426k",
                model="google/gemma-2-2b",
                repo_id="mntss/clt-gemma-2-2b-426k",
                layers=None,  # Cross-layer
                hook_point_type="mlp",
                d_sae=426_000,
                expansion_factor=None,
                description="Cross-layer transcoders for Gemma 2 2B (426K features)",
                source="circuit-tracer",
            ),
            SAEInfo(
                name="gemma-2-2b-clt-2.5m",
                model="google/gemma-2-2b",
                repo_id="mntss/clt-gemma-2-2b-2.5M",
                layers=None,
                hook_point_type="mlp",
                d_sae=2_500_000,
                expansion_factor=None,
                description="Cross-layer transcoders for Gemma 2 2B (2.5M features)",
                source="circuit-tracer",
            ),
            SAEInfo(
                name="llama-3.2-1b-plt",
                model="meta-llama/Llama-3.2-1B",
                repo_id="mntss/transcoder-Llama-3.2-1B",
                layers=list(range(16)),  # Llama 3.2 1B has 16 layers
                hook_point_type="mlp",
                d_sae=None,
                expansion_factor=None,
                description="Per-layer transcoders for Llama 3.2 1B",
                source="circuit-tracer",
            ),
            SAEInfo(
                name="llama-3.2-1b-clt-524k",
                model="meta-llama/Llama-3.2-1B",
                repo_id="mntss/clt-llama-3.2-1b-524k",
                layers=None,
                hook_point_type="mlp",
                d_sae=524_000,
                expansion_factor=None,
                description="Cross-layer transcoders for Llama 3.2 1B (524K features)",
                source="circuit-tracer",
            ),
            # Qwen models
            SAEInfo(
                name="qwen3-0.6b-plt",
                model="Qwen/Qwen3-0.6B",
                repo_id="mwhanna/qwen3-0.6b-transcoders-lowl0",
                layers=None,
                hook_point_type="mlp",
                d_sae=None,
                expansion_factor=None,
                description="Per-layer transcoders for Qwen 3 0.6B",
                source="circuit-tracer",
            ),
            SAEInfo(
                name="qwen3-1.7b-plt",
                model="Qwen/Qwen3-1.7B",
                repo_id="mwhanna/qwen3-1.7b-transcoders-lowl0",
                layers=None,
                hook_point_type="mlp",
                d_sae=None,
                expansion_factor=None,
                description="Per-layer transcoders for Qwen 3 1.7B",
                source="circuit-tracer",
            ),
        ]

        for sae_info in native_transcoders:
            self.circuit_tracer_saes[sae_info.name] = sae_info

    def list_available_models(self) -> list[str]:
        """Get list of all models with available SAEs.

        Returns:
            List of unique model names
        """
        models = set()

        # From circuit-tracer native
        for sae in self.circuit_tracer_saes.values():
            models.add(sae.model)

        # From SAELens (would need to parse the registry structure)
        # The registry is complex, so we'll add known ones
        models.update([
            "gpt2-small",
            "meta-llama/Llama-3-8B",
            "meta-llama/Llama-3.1-8B",
            "mistralai/Mistral-7B-v0.1",
            "microsoft/phi-3-mini",
            "google/gemma-2-9b",
        ])

        return sorted(list(models))

    def search_saes(
        self,
        model_name: str | None = None,
        hook_type: str | None = None,
        min_features: int | None = None,
    ) -> list[SAEInfo]:
        """Search for available SAEs matching criteria.

        Args:
            model_name: Filter by model name (partial match)
            hook_type: Filter by hook point type (e.g., "mlp", "resid", "attn")
            min_features: Minimum number of features

        Returns:
            List of matching SAEInfo objects
        """
        results = []

        for sae in self.circuit_tracer_saes.values():
            # Apply filters
            if model_name and model_name.lower() not in sae.model.lower():
                continue
            if hook_type and hook_type.lower() not in sae.hook_point_type.lower():
                continue
            if min_features and sae.d_sae and sae.d_sae < min_features:
                continue

            results.append(sae)

        return results

    def get_sae_info(self, name_or_repo: str) -> SAEInfo | None:
        """Get information about a specific SAE.

        Args:
            name_or_repo: SAE name or HuggingFace repo ID

        Returns:
            SAEInfo if found, None otherwise
        """
        # Try by name first
        if name_or_repo in self.circuit_tracer_saes:
            return self.circuit_tracer_saes[name_or_repo]

        # Try by repo_id
        for sae in self.circuit_tracer_saes.values():
            if sae.repo_id == name_or_repo:
                return sae

        return None

    def print_summary(self):
        """Print a summary of available SAEs."""
        print("=" * 70)
        print("AVAILABLE SAES/TRANSCODERS FOR CIRCUIT-TRACER")
        print("=" * 70)

        print(f"\nCircuit-Tracer Native Transcoders: {len(self.circuit_tracer_saes)}")
        print("-" * 70)

        # Group by model
        by_model: dict[str, list[SAEInfo]] = {}
        for sae in self.circuit_tracer_saes.values():
            if sae.model not in by_model:
                by_model[sae.model] = []
            by_model[sae.model].append(sae)

        for model, saes in sorted(by_model.items()):
            print(f"\n{model}:")
            for sae in saes:
                features_str = f"{sae.d_sae:,}" if sae.d_sae else "variable"
                print(f"  - {sae.name}")
                print(f"    Repo: {sae.repo_id}")
                print(f"    Features: {features_str}")
                print(f"    {sae.description}")

        print("\n" + "=" * 70)
        print(f"Total models supported: {len(self.list_available_models())}")
        print("=" * 70)

        print("\nAdditional SAEs available via SAELens:")
        print("  - GPT-2 Small")
        print("  - Llama 3 / 3.1 (8B)")
        print("  - Mistral 7B")
        print("  - Phi-3 Mini")
        print("  - Gemma 2 (9B)")
        print("\nTo use SAELens SAEs, they may require format conversion.")
        print("=" * 70)


def discover_saes(model_name: str | None = None) -> list[SAEInfo]:
    """Convenience function to discover available SAEs.

    Args:
        model_name: Optional model name to filter by

    Returns:
        List of available SAEs
    """
    registry = SAERegistry()

    if model_name:
        return registry.search_saes(model_name=model_name)
    else:
        return list(registry.circuit_tracer_saes.values())


if __name__ == "__main__":
    # CLI usage
    import sys

    registry = SAERegistry()

    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        print(f"Searching for SAEs for: {model_name}\n")
        results = registry.search_saes(model_name=model_name)
        if results:
            for sae in results:
                print(f"Name: {sae.name}")
                print(f"Repo: {sae.repo_id}")
                print(f"Description: {sae.description}")
                print()
        else:
            print(f"No SAEs found for model: {model_name}")
    else:
        registry.print_summary()
