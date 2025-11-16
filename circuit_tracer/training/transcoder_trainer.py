"""Wrapper for training transcoders using external libraries (EleutherAI Sparsify, SAELens)."""

import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

logger = logging.getLogger(__name__)


@dataclass
class TranscoderTrainingConfig:
    """Configuration for transcoder training.

    This configuration is used to train transcoders using EleutherAI Sparsify
    or other training libraries, then export them in circuit-tracer format.
    """

    model_name: str
    """HuggingFace model name (e.g., 'google/gemma-2-2b')"""

    dataset: str = "c4"
    """Dataset to train on (e.g., 'c4', 'openwebtext', or HF dataset name)"""

    d_transcoder: int = 65536
    """Number of transcoder features (typically 4-32x model dimension)"""

    n_tokens: int = 100_000_000
    """Number of training tokens"""

    sparsity_coef: float = 0.01
    """L1 sparsity coefficient"""

    learning_rate: float = 1e-4
    """Learning rate for training"""

    batch_size: int = 32
    """Training batch size"""

    use_transcoders: bool = True
    """If True, train transcoders (input->output). If False, train SAEs (point activations)"""

    output_dir: str = "./trained_transcoders"
    """Directory to save trained weights"""

    layers: list[int] | None = None
    """Which layers to train transcoders for. If None, trains for all layers"""

    device: str = "cuda"
    """Device to train on"""

    save_interval: int = 10_000
    """Save checkpoint every N steps"""

    eval_interval: int = 5_000
    """Evaluate reconstruction loss every N steps"""

    wandb_project: str | None = None
    """Weights & Biases project name for logging (optional)"""

    extra_args: dict[str, any] = field(default_factory=dict)
    """Additional arguments to pass to the training library"""


class TranscoderTrainer:
    """Wrapper for training transcoders using external libraries."""

    def __init__(self, config: TranscoderTrainingConfig):
        """Initialize trainer with configuration.

        Args:
            config: Training configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def check_dependencies(self) -> tuple[bool, list[str]]:
        """Check if required training libraries are installed.

        Returns:
            Tuple of (all_installed, missing_packages)
        """
        missing = []

        # Check for EleutherAI Sparsify
        try:
            import sparsify  # noqa: F401
        except ImportError:
            missing.append("sparsify")

        return len(missing) == 0, missing

    def train_with_sparsify(self, verbose: bool = True) -> Path:
        """Train transcoders using EleutherAI Sparsify.

        This uses the sparsify library to train k-sparse autoencoders or transcoders.

        Args:
            verbose: Whether to print training progress

        Returns:
            Path to the trained model directory

        Raises:
            RuntimeError: If sparsify is not installed or training fails
        """
        deps_ok, missing = self.check_dependencies()
        if not deps_ok:
            raise RuntimeError(
                f"Missing dependencies: {missing}. "
                "Install with: pip install eleutherai-sparsify"
            )

        logger.info(f"Training transcoders for {self.config.model_name}...")
        logger.info(f"Config: {self.config}")

        # Build sparsify command
        cmd = [
            sys.executable,
            "-m",
            "sparsify",
            self.config.model_name,
            self.config.dataset,
        ]

        if self.config.use_transcoders:
            cmd.append("--transcode")

        # Add configuration arguments
        cmd.extend(
            [
                "--d_hidden",
                str(self.config.d_transcoder),
                "--lr",
                str(self.config.learning_rate),
                "--batch_size",
                str(self.config.batch_size),
                "--n_tokens",
                str(self.config.n_tokens),
                "--sparsity_coef",
                str(self.config.sparsity_coef),
                "--save_dir",
                str(self.output_dir),
                "--device",
                self.config.device,
            ]
        )

        if self.config.layers:
            cmd.extend(["--layers"] + [str(l) for l in self.config.layers])

        if self.config.wandb_project:
            cmd.extend(["--wandb_project", self.config.wandb_project])

        # Add any extra arguments
        for key, value in self.config.extra_args.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])

        logger.info(f"Running command: {' '.join(cmd)}")

        try:
            if verbose:
                # Stream output in real-time
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
                for line in process.stdout:
                    print(line, end="")
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError(f"Training failed with exit code {process.returncode}")
            else:
                # Capture output
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                logger.info(result.stdout)

            logger.info(f"Training complete! Models saved to {self.output_dir}")
            return self.output_dir

        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e}")
            logger.error(f"Output: {e.output}")
            raise RuntimeError(f"Sparsify training failed: {e}")

    def train_with_saelens(self, verbose: bool = True) -> Path:
        """Train SAEs using SAELens library.

        Note: This is a placeholder. SAELens has its own API and this would need
        to be implemented based on their documentation.

        Args:
            verbose: Whether to print training progress

        Returns:
            Path to the trained model directory

        Raises:
            NotImplementedError: SAELens integration not yet implemented
        """
        raise NotImplementedError(
            "SAELens integration not yet implemented. "
            "Please use train_with_sparsify() or train SAEs directly with SAELens "
            "and then use export_for_circuit_tracer() to convert the format."
        )

    def train(
        self, method: Literal["sparsify", "saelens"] = "sparsify", verbose: bool = True
    ) -> Path:
        """Train transcoders using the specified method.

        Args:
            method: Which training library to use
            verbose: Whether to print training progress

        Returns:
            Path to the trained model directory
        """
        if method == "sparsify":
            return self.train_with_sparsify(verbose=verbose)
        elif method == "saelens":
            return self.train_with_saelens(verbose=verbose)
        else:
            raise ValueError(f"Unknown training method: {method}")


def export_for_circuit_tracer(
    transcoder_dir: str | Path,
    model_name: str,
    output_dir: str | Path,
    transcoder_type: Literal["plt", "clt"] = "plt",
    feature_input_hook: str | None = None,
    feature_output_hook: str | None = None,
) -> Path:
    """Export trained transcoders in circuit-tracer compatible format.

    This creates a config.yaml file and organizes the transcoder weights
    in the format expected by circuit-tracer's HuggingFace loading utilities.

    Args:
        transcoder_dir: Directory containing trained transcoder weights
        model_name: Name of the base model these transcoders were trained on
        output_dir: Where to save the circuit-tracer compatible format
        transcoder_type: Type of transcoder ("plt" for per-layer, "clt" for cross-layer)
        feature_input_hook: Hook name for feature inputs (auto-inferred if None)
        feature_output_hook: Hook name for feature outputs (auto-inferred if None)

    Returns:
        Path to the exported directory
    """
    transcoder_dir = Path(transcoder_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-infer hook names based on model if not provided
    if feature_input_hook is None:
        # Default hook for MLP input
        feature_input_hook = "blocks.{layer}.hook_mlp_in"

    if feature_output_hook is None:
        # Default hook for MLP output
        feature_output_hook = "blocks.{layer}.hook_mlp_out"

    # Create config.yaml
    config = {
        "model_name": model_name,
        "model_kind": "transcoder_set" if transcoder_type == "plt" else "cross_layer_transcoder",
        "feature_input_hook": feature_input_hook,
        "feature_output_hook": feature_output_hook,
        "scan": f"custom/{model_name.replace('/', '_')}",
    }

    # If PLT, list individual layer files
    if transcoder_type == "plt":
        layer_files = sorted(transcoder_dir.glob("layer_*.safetensors"))
        if not layer_files:
            logger.warning(f"No layer_*.safetensors files found in {transcoder_dir}")
            # Try to find any .safetensors files
            layer_files = sorted(transcoder_dir.glob("*.safetensors"))

        config["transcoders"] = [str(f) for f in layer_files]

    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    logger.info(f"Created config.yaml at {config_path}")

    # Copy or symlink transcoder files
    import shutil

    for src_file in transcoder_dir.glob("*.safetensors"):
        dst_file = output_dir / src_file.name
        if not dst_file.exists():
            shutil.copy2(src_file, dst_file)
            logger.info(f"Copied {src_file.name} to {output_dir}")

    logger.info(f"Transcoders exported to {output_dir} in circuit-tracer format")
    logger.info(
        f"You can now use this with circuit-tracer by pointing to: {output_dir.absolute()}"
    )

    return output_dir
