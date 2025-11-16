"""Task configuration management for evaluation."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ExpectedFeature:
    """Specification for an expected feature pattern in a task."""

    type: str  # "pattern", "layer_position", "semantic"
    name: str
    description: str
    matcher: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricConfig:
    """Configuration for an evaluation metric."""

    weight: float = 1.0
    description: str = ""
    enabled: bool = True


@dataclass
class InterventionTest:
    """Specification for an intervention test."""

    name: str
    description: str
    expected_change: str  # "high", "medium", "low"
    intervention_type: str = "ablate"  # "ablate", "amplify", "swap"


@dataclass
class DatasetInfo:
    """Information about a task dataset."""

    name: str
    path: str
    size: int
    description: str = ""


@dataclass
class TaskConfig:
    """Complete configuration for a task evaluation."""

    task_name: str
    description: str
    expected_features: list[ExpectedFeature] = field(default_factory=list)
    evaluation_metrics: dict[str, MetricConfig] = field(default_factory=dict)
    intervention_tests: list[InterventionTest] = field(default_factory=list)
    datasets: list[DatasetInfo] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskConfig":
        """Create TaskConfig from a dictionary."""
        # Parse expected features
        expected_features = []
        for feat_data in data.get("expected_features", []):
            expected_features.append(
                ExpectedFeature(
                    type=feat_data["type"],
                    name=feat_data["name"],
                    description=feat_data["description"],
                    matcher=feat_data.get("matcher", {}),
                )
            )

        # Parse evaluation metrics
        evaluation_metrics = {}
        for metric_name, metric_data in data.get("evaluation_metrics", {}).items():
            evaluation_metrics[metric_name] = MetricConfig(
                weight=metric_data.get("weight", 1.0),
                description=metric_data.get("description", ""),
                enabled=metric_data.get("enabled", True),
            )

        # Parse intervention tests
        intervention_tests = []
        for test_data in data.get("intervention_tests", []):
            intervention_tests.append(
                InterventionTest(
                    name=test_data["name"],
                    description=test_data["description"],
                    expected_change=test_data["expected_change"],
                    intervention_type=test_data.get("intervention_type", "ablate"),
                )
            )

        # Parse datasets
        datasets = []
        for ds_data in data.get("datasets", []):
            datasets.append(
                DatasetInfo(
                    name=ds_data["name"],
                    path=ds_data["path"],
                    size=ds_data["size"],
                    description=ds_data.get("description", ""),
                )
            )

        return cls(
            task_name=data["task_name"],
            description=data["description"],
            expected_features=expected_features,
            evaluation_metrics=evaluation_metrics,
            intervention_tests=intervention_tests,
            datasets=datasets,
            metadata=data.get("metadata", {}),
        )


def load_task_config(config_path: str | Path) -> TaskConfig:
    """Load a task configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        TaskConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Task configuration not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    logger.info(f"Loaded task configuration: {data.get('task_name', 'unknown')}")

    return TaskConfig.from_dict(data)


def save_task_config(config: TaskConfig, output_path: str | Path) -> None:
    """Save a task configuration to a YAML file.

    Args:
        config: TaskConfig object to save
        output_path: Path where to save the YAML file
    """
    output_path = Path(output_path)

    # Convert to dictionary
    data = {
        "task_name": config.task_name,
        "description": config.description,
        "expected_features": [
            {
                "type": feat.type,
                "name": feat.name,
                "description": feat.description,
                "matcher": feat.matcher,
            }
            for feat in config.expected_features
        ],
        "evaluation_metrics": {
            name: {
                "weight": metric.weight,
                "description": metric.description,
                "enabled": metric.enabled,
            }
            for name, metric in config.evaluation_metrics.items()
        },
        "intervention_tests": [
            {
                "name": test.name,
                "description": test.description,
                "expected_change": test.expected_change,
                "intervention_type": test.intervention_type,
            }
            for test in config.intervention_tests
        ],
        "datasets": [
            {
                "name": ds.name,
                "path": ds.path,
                "size": ds.size,
                "description": ds.description,
            }
            for ds in config.datasets
        ],
        "metadata": config.metadata,
    }

    with open(output_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved task configuration to: {output_path}")
