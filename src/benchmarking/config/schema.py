"""Durable config schema for benchmark train/eval runners."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

from src.benchmarking.metrics.units import canonicalize_point_unit
from src.benchmarking.registry.model_registry import ModelRegistry

DistanceMode = Literal["visible_only", "overlap_only", "visible_overlap_preferred"]
TrainMode = Literal["smoke", "full"]
SamplingMode = Literal["none", "random", "voxel", "fps"]
NoiseApplyMode = Literal["source", "both"]


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str = "c3vd_raycasting_v1"
    split: str = "test"
    subset_name: str | None = None
    point_unit: str = "m"
    official_track: str = "main"


@dataclass(frozen=True)
class DataConfig:
    manifest_path: str
    subset_config_path: str | None = None
    dataset_root: str | None = None


@dataclass(frozen=True)
class PreprocessConfig:
    profile: str = "canonical_v1"
    seed: int = 42
    sampling_override: SamplingMode | None = None
    num_points_override: int | None = None


@dataclass(frozen=True)
class PerturbationConfig:
    enabled: bool = False
    rotation_deg: float = 0.0
    translation_m: float = 0.0
    min_rotation_deg: float = 0.0
    min_translation_m: float = 0.0
    noise_sigma: float = 0.0
    noise_clip: float = 0.0
    apply_noise_to: NoiseApplyMode = "source"


@dataclass(frozen=True)
class ModelConfig:
    id: str
    checkpoint_path: str | None = None
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cuda:0"
    batch_size: int = 1
    num_workers: int = 0
    export_html: bool = True
    output_dir: str = "outputs/benchmark_eval"
    train_mode: TrainMode = "smoke"
    train_metadata_path: str | None = None
    dataset_overrides: dict[str, Any] = field(default_factory=dict)
    training_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisGeometryConfig:
    sample_count: int = 2048
    distance_mode: DistanceMode = "visible_overlap_preferred"
    export_histogram: bool = True
    export_cdf: bool = True


@dataclass(frozen=True)
class AnalysisQualitativeConfig:
    topk_failures: int = 20
    export_failure_gallery: bool = True


@dataclass(frozen=True)
class AnalysisExportConfig:
    html: bool = True
    png: bool = True
    markdown_tables: bool = True


@dataclass(frozen=True)
class AnalysisConfig:
    required_tables: list[str] = field(
        default_factory=lambda: [
            "leaderboard_main",
            "leaderboard_multithreshold",
            "bucket_overlap",
            "bucket_rotation",
            "bucket_scene",
            "efficiency_summary",
            "geometry_summary",
        ]
    )
    required_curves: list[str] = field(
        default_factory=lambda: ["rr_multithreshold", "success_latency_pareto"]
    )
    bucket_keys: list[str] = field(
        default_factory=lambda: [
            "overlap_bin",
            "rotation_bin",
            "translation_bin",
            "scene_id",
            "preprocess_profile_id",
            "refinement_track",
        ]
    )
    geometry: AnalysisGeometryConfig = field(default_factory=AnalysisGeometryConfig)
    qualitative: AnalysisQualitativeConfig = field(
        default_factory=AnalysisQualitativeConfig
    )
    export: AnalysisExportConfig = field(default_factory=AnalysisExportConfig)


@dataclass(frozen=True)
class ExperimentConfig:
    benchmark: BenchmarkConfig
    data: DataConfig
    preprocess: PreprocessConfig
    model: ModelConfig
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"Config section '{key}' must be a mapping.")
    return value


def _merge(defaults: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(defaults.get(key), Mapping):
            merged[key] = _merge(defaults[key], value)
        else:
            merged[key] = value
    return merged


def validate_config(config: Mapping[str, Any]) -> ExperimentConfig:
    """Validate a loaded config mapping and return the normalized dataclass."""

    benchmark_payload = _merge(
        asdict(BenchmarkConfig()), _require_mapping(config, "benchmark")
    )
    benchmark_payload["point_unit"] = canonicalize_point_unit(
        str(benchmark_payload["point_unit"])
    )
    benchmark = BenchmarkConfig(**benchmark_payload)
    data_section = _require_mapping(config, "data")
    model_section = _require_mapping(config, "model")

    if "manifest_path" not in data_section:
        raise KeyError("Config section 'data' requires 'manifest_path'.")
    if "id" not in model_section:
        raise KeyError("Config section 'model' requires 'id'.")

    preprocess = PreprocessConfig(
        **_merge(asdict(PreprocessConfig()), _require_mapping(config, "preprocess"))
    )
    if preprocess.sampling_override not in {None, "none", "random", "voxel", "fps"}:
        raise ValueError(
            "Config section 'preprocess.sampling_override' must be one of "
            "'none', 'random', 'voxel', 'fps', or null."
        )
    if (
        preprocess.num_points_override is not None
        and int(preprocess.num_points_override) <= 0
    ):
        raise ValueError(
            "Config section 'preprocess.num_points_override' must be positive "
            "when provided."
        )
    perturbation = PerturbationConfig(
        **_merge(
            asdict(PerturbationConfig()),
            _require_mapping(config, "perturbation"),
        )
    )
    if perturbation.apply_noise_to not in {"source", "both"}:
        raise ValueError(
            "Config section 'perturbation.apply_noise_to' must be either "
            "'source' or 'both'."
        )
    for key in (
        "rotation_deg",
        "translation_m",
        "min_rotation_deg",
        "min_translation_m",
        "noise_sigma",
        "noise_clip",
    ):
        value = float(getattr(perturbation, key))
        if value < 0.0:
            raise ValueError(
                f"Config section 'perturbation.{key}' must be non-negative."
            )
    if perturbation.min_rotation_deg > perturbation.rotation_deg:
        raise ValueError(
            "Config section 'perturbation.min_rotation_deg' must be <= "
            "'perturbation.rotation_deg'."
        )
    if perturbation.min_translation_m > perturbation.translation_m:
        raise ValueError(
            "Config section 'perturbation.min_translation_m' must be <= "
            "'perturbation.translation_m'."
        )
    if perturbation.noise_clip > 0.0 and perturbation.noise_sigma <= 0.0:
        raise ValueError(
            "Config section 'perturbation.noise_clip' requires "
            "'perturbation.noise_sigma' to be > 0."
        )
    registry = ModelRegistry()
    spec = registry.get(str(model_section["id"]))
    model_payload = _merge(asdict(ModelConfig(id=model_section["id"])), model_section)
    raw_overrides = model_payload.get("overrides")
    if raw_overrides is None:
        raw_overrides = {}
    if not isinstance(raw_overrides, Mapping):
        raise TypeError("Config section 'model.overrides' must be a mapping.")
    model_payload["overrides"] = registry.apply_default_model_overrides(
        model_id=str(model_section["id"]),
        overrides=dict(raw_overrides),
    )
    normalize_mode = model_payload["overrides"].get("normalize_mode")
    if normalize_mode is not None and normalize_mode not in {
        "none",
        "unit_cube",
        "joint",
    }:
        raise ValueError(
            "Config section 'model.overrides.normalize_mode' must be one of "
            "'none', 'unit_cube', 'joint', or null."
        )
    model = ModelConfig(**model_payload)
    runtime = RuntimeConfig(
        **_merge(asdict(RuntimeConfig()), _require_mapping(config, "runtime"))
    )
    if not isinstance(runtime.dataset_overrides, Mapping):
        raise TypeError(
            "Config section 'runtime.dataset_overrides' must be a mapping."
        )
    allowed_dataset_overrides = {
        "frame_stride",
        "max_train_pairs",
        "max_val_pairs",
        "max_test_pairs",
        "num_points",
        "sampling_mode",
    }
    unsupported_dataset_overrides = sorted(
        set(runtime.dataset_overrides) - allowed_dataset_overrides
    )
    if unsupported_dataset_overrides:
        raise ValueError(
            "Unsupported runtime.dataset_overrides keys: "
            + ", ".join(unsupported_dataset_overrides)
        )
    for key in (
        "frame_stride",
        "max_train_pairs",
        "max_val_pairs",
        "max_test_pairs",
        "num_points",
    ):
        value = runtime.dataset_overrides.get(key)
        if value is None and key in runtime.dataset_overrides and key not in {
            "max_train_pairs",
            "max_val_pairs",
            "max_test_pairs",
        }:
            raise ValueError(
                f"Config section 'runtime.dataset_overrides.{key}' cannot be null."
            )
        if value is not None and int(value) <= 0:
            raise ValueError(
                f"Config section 'runtime.dataset_overrides.{key}' must be "
                "positive when provided."
            )
    sampling_mode_override = runtime.dataset_overrides.get("sampling_mode")
    if sampling_mode_override not in {None, "none", "random", "voxel", "fps"}:
        raise ValueError(
            "Config section 'runtime.dataset_overrides.sampling_mode' must be one "
            "of 'none', 'random', 'voxel', 'fps', or null."
        )
    if not isinstance(runtime.training_overrides, Mapping):
        raise TypeError(
            "Config section 'runtime.training_overrides' must be a mapping."
        )
    if runtime.train_mode not in {"smoke", "full"}:
        raise ValueError(
            "Config section 'runtime.train_mode' must be one of "
            "'smoke' or 'full'."
        )
    runtime_device = str(runtime.device)
    if spec.runtime_policy == "cpu_only" and runtime_device != "cpu":
        raise ValueError(
            f"Config section 'runtime.device' must be 'cpu' for model "
            f"'{model.id}'. Received {runtime_device!r}."
        )
    if spec.runtime_policy == "cuda_required" and not runtime_device.startswith(
        "cuda"
    ):
        raise ValueError(
            f"Config section 'runtime.device' must be a CUDA device such as "
            f"'cuda:0' for model '{model.id}'. Received {runtime_device!r}."
        )

    analysis_defaults = asdict(AnalysisConfig())
    analysis_merged = _merge(analysis_defaults, _require_mapping(config, "analysis"))
    geometry = AnalysisGeometryConfig(**analysis_merged["geometry"])
    qualitative = AnalysisQualitativeConfig(**analysis_merged["qualitative"])
    export = AnalysisExportConfig(**analysis_merged["export"])
    analysis = AnalysisConfig(
        required_tables=list(analysis_merged["required_tables"]),
        required_curves=list(analysis_merged["required_curves"]),
        bucket_keys=list(analysis_merged["bucket_keys"]),
        geometry=geometry,
        qualitative=qualitative,
        export=export,
    )
    data = DataConfig(
        **_merge(
            asdict(DataConfig(manifest_path=data_section["manifest_path"])),
            data_section,
        )
    )

    required_tables = set(analysis.required_tables)
    if "leaderboard_main" not in required_tables:
        raise ValueError("analysis.required_tables must include 'leaderboard_main'.")
    if "leaderboard_multithreshold" not in required_tables:
        raise ValueError(
            "analysis.required_tables must include 'leaderboard_multithreshold'."
        )
    if geometry.distance_mode not in {
        "visible_only",
        "overlap_only",
        "visible_overlap_preferred",
    }:
        raise ValueError(
            "analysis.geometry.distance_mode must be one of "
            "'visible_only', 'overlap_only', 'visible_overlap_preferred'."
        )
    if (
        benchmark.official_track == "main"
        and preprocess.profile == "legacy_joint_norm_v1"
    ):
        raise ValueError(
            "legacy_joint_norm_v1 belongs to the legacy track "
            "and cannot be used as main."
        )

    if not runtime.export_html:
        analysis = AnalysisConfig(
            required_tables=analysis.required_tables,
            required_curves=analysis.required_curves,
            bucket_keys=analysis.bucket_keys,
            geometry=analysis.geometry,
            qualitative=analysis.qualitative,
            export=AnalysisExportConfig(
                html=False,
                png=analysis.export.png,
                markdown_tables=analysis.export.markdown_tables,
            ),
        )

    return ExperimentConfig(
        benchmark=benchmark,
        data=data,
        preprocess=preprocess,
        perturbation=perturbation,
        model=model,
        runtime=runtime,
        analysis=analysis,
    )
