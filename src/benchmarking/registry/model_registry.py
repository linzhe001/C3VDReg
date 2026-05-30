"""Model registry and capability contract for benchmark adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SourceKind = Literal["benchmark_native", "vendor_readonly"]
RuntimePolicy = Literal["cpu_only", "cuda_required"]
NormalizeMode = Literal["none", "unit_cube", "joint"]


class UnavailableAdapter:
    """Explicit placeholder for model families without a benchmark eval adapter."""

    def __init__(self, *_args, **_kwargs) -> None:
        raise RuntimeError(
            "This model family does not yet have a benchmark eval adapter. "
            "Do not fall back to a different adapter silently."
        )


@dataclass(frozen=True)
class AdapterCapabilities:
    supports_train: bool
    supports_eval: bool
    accepts_normals: bool
    accepts_variable_points: bool
    allowed_preprocess_profiles: tuple[str, ...]
    private_input_transform_id: str
    refinement_mode: str


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    source_kind: SourceKind
    runtime_policy: RuntimePolicy
    adapter_path: str
    train_bridge: str | None
    eval_bridge: str
    checkpoint_path: str | None
    baseline_repo_path: str | None
    capabilities: AdapterCapabilities
    baseline_normalization_policy_id: str = "raw_metric"
    baseline_normalization_reference: str = "untracked"
    default_eval_normalize_mode: NormalizeMode | None = None


class ModelRegistry:
    """Registry of model specs used by train/eval runners."""

    def __init__(self, register_defaults: bool = True) -> None:
        self._specs: dict[str, ModelSpec] = {}
        if register_defaults:
            for spec in _default_specs():
                self.register(spec)

    def register(self, spec: ModelSpec) -> None:
        self._specs[spec.model_id] = spec

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self._specs[model_id]
        except KeyError as exc:
            raise KeyError(f"Unknown model_id '{model_id}'.") from exc

    def assert_compatible(self, model_id: str, preprocess_profile_id: str) -> None:
        spec = self.get(model_id)
        if preprocess_profile_id not in spec.capabilities.allowed_preprocess_profiles:
            allowed = ", ".join(spec.capabilities.allowed_preprocess_profiles)
            raise ValueError(
                f"Model '{model_id}' does not allow preprocess profile "
                f"'{preprocess_profile_id}'. Allowed: {allowed}"
            )
        if (
            preprocess_profile_id == "normals_v1"
            and not spec.capabilities.accepts_normals
        ):
            raise ValueError(f"Model '{model_id}' does not accept normals_v1 inputs.")

    def default_model_overrides(self, model_id: str) -> dict[str, object]:
        spec = self.get(model_id)
        overrides: dict[str, object] = {}
        if spec.default_eval_normalize_mode is not None:
            overrides["normalize_mode"] = spec.default_eval_normalize_mode
        return overrides

    def apply_default_model_overrides(
        self,
        model_id: str,
        overrides: dict[str, object] | None,
    ) -> dict[str, object]:
        merged = self.default_model_overrides(model_id)
        if overrides:
            merged.update(dict(overrides))
        return merged


def _default_specs() -> tuple[ModelSpec, ...]:
    legacy_profiles = ("canonical_v1", "legacy_joint_norm_v1", "debug_raw_v1")
    main_profiles = ("canonical_v1", "debug_raw_v1")
    return (
        ModelSpec(
            model_id="icp",
            family="classical",
            source_kind="benchmark_native",
            runtime_policy="cpu_only",
            adapter_path="src.unified_testing.adapters.icp_adapter.ICPAdapter",
            train_bridge=None,
            eval_bridge="src.unified_testing.unified_test",
            checkpoint_path=None,
            baseline_repo_path=None,
            capabilities=AdapterCapabilities(
                supports_train=False,
                supports_eval=True,
                accepts_normals=False,
                accepts_variable_points=True,
                allowed_preprocess_profiles=legacy_profiles,
                private_input_transform_id="bn3_metric_unit",
                refinement_mode="none",
            ),
            baseline_normalization_policy_id="metric_scene_raw_no_global_norm",
            baseline_normalization_reference="classical_icp_metric_scene_input",
            default_eval_normalize_mode="none",
        ),
        ModelSpec(
            model_id="dcp",
            family="learning",
            source_kind="vendor_readonly",
            runtime_policy="cuda_required",
            adapter_path="src.unified_testing.adapters.dcp_adapter.DCPAdapter",
            train_bridge="src.benchmarking.bridges.train_dcp_c3vd",
            eval_bridge="src.unified_testing.unified_test",
            checkpoint_path=None,
            baseline_repo_path="baselines/dcp",
            capabilities=AdapterCapabilities(
                supports_train=True,
                supports_eval=True,
                accepts_normals=False,
                accepts_variable_points=False,
                allowed_preprocess_profiles=legacy_profiles,
                private_input_transform_id="b3n_modelnet_unit_cube_shared",
                refinement_mode="none",
            ),
            baseline_normalization_policy_id="modelnet_hdf5_pre_normalized_unit_scale",
            baseline_normalization_reference="baselines/dcp/data.py",
            default_eval_normalize_mode="unit_cube",
        ),
        ModelSpec(
            model_id="pointnetlk",
            family="learning",
            source_kind="vendor_readonly",
            runtime_policy="cuda_required",
            adapter_path="src.unified_testing.adapters.pointnetlk_adapter.PointNetLKAdapter",
            train_bridge="src.benchmarking.bridges.train_pointnetlk_c3vd",
            eval_bridge="src.unified_testing.unified_test",
            checkpoint_path=None,
            baseline_repo_path="baselines/PointNetLK",
            capabilities=AdapterCapabilities(
                supports_train=True,
                supports_eval=True,
                accepts_normals=False,
                accepts_variable_points=False,
                allowed_preprocess_profiles=legacy_profiles,
                private_input_transform_id="bn3_unit_cube_zero_mean",
                refinement_mode="none",
            ),
            baseline_normalization_policy_id="modelnet_unit_cube_zero_mean",
            baseline_normalization_reference="baselines/PointNetLK/ptlk/data/transforms.py",
            default_eval_normalize_mode="unit_cube",
        ),
        ModelSpec(
            model_id="pointnetlk_revisited",
            family="learning",
            source_kind="vendor_readonly",
            runtime_policy="cuda_required",
            adapter_path=(
                "src.unified_testing.adapters.pointnetlk_revisited_adapter."
                "PointNetLKRevisitedAdapter"
            ),
            train_bridge="src.benchmarking.bridges.train_pointnetlk_revisited_c3vd",
            eval_bridge="src.unified_testing.unified_test",
            checkpoint_path=None,
            baseline_repo_path="baselines/PointNetLK_Revisited",
            capabilities=AdapterCapabilities(
                supports_train=True,
                supports_eval=True,
                accepts_normals=False,
                accepts_variable_points=False,
                allowed_preprocess_profiles=legacy_profiles,
                private_input_transform_id="bn3_metric_unit_no_global_norm",
                refinement_mode="none",
            ),
            baseline_normalization_policy_id="threedmatch_voxel_metric_raw_no_global_norm",
            baseline_normalization_reference="baselines/PointNetLK_Revisited/data_utils.py",
            default_eval_normalize_mode="none",
        ),
        ModelSpec(
            model_id="mamba3d",
            family="learning",
            source_kind="benchmark_native",
            runtime_policy="cuda_required",
            adapter_path=(
                "src.unified_testing.adapters.pointnetlk_c3vd_adapter."
                "PointNetLKC3VDAdapter"
            ),
            train_bridge="src.benchmarking.bridges.train_mamba3d_c3vd",
            eval_bridge="src.unified_testing.unified_test",
            checkpoint_path=None,
            baseline_repo_path=None,
            capabilities=AdapterCapabilities(
                supports_train=True,
                supports_eval=True,
                accepts_normals=False,
                accepts_variable_points=False,
                allowed_preprocess_profiles=legacy_profiles,
                private_input_transform_id="bn3_c3vd_unit_cube_shared",
                refinement_mode="none",
            ),
            baseline_normalization_policy_id="c3vd_pointnetlk_style_unit_scale",
            baseline_normalization_reference=(
                "forks/PointNetLK_c3vd/ptlk/data/datasets.py"
            ),
            default_eval_normalize_mode="unit_cube",
        ),
        ModelSpec(
            model_id="mamba3d_true",
            family="learning",
            source_kind="benchmark_native",
            runtime_policy="cuda_required",
            adapter_path=(
                "src.unified_testing.adapters.pointnetlk_c3vd_adapter."
                "PointNetLKC3VDAdapter"
            ),
            train_bridge="src.benchmarking.bridges.train_mamba3d_c3vd",
            eval_bridge="src.unified_testing.unified_test",
            checkpoint_path=None,
            baseline_repo_path=None,
            capabilities=AdapterCapabilities(
                supports_train=True,
                supports_eval=True,
                accepts_normals=False,
                accepts_variable_points=False,
                allowed_preprocess_profiles=legacy_profiles,
                private_input_transform_id="bn3_c3vd_unit_cube_shared",
                refinement_mode="none",
            ),
            baseline_normalization_policy_id="c3vd_pointnetlk_style_unit_scale",
            baseline_normalization_reference=(
                "forks/PointNetLK_c3vd/ptlk/data/datasets.py"
            ),
            default_eval_normalize_mode="unit_cube",
        ),
        ModelSpec(
            model_id="mamba3d_mamba2",
            family="learning",
            source_kind="benchmark_native",
            runtime_policy="cuda_required",
            adapter_path=(
                "src.unified_testing.adapters.pointnetlk_c3vd_adapter."
                "PointNetLKC3VDAdapter"
            ),
            train_bridge="src.benchmarking.bridges.train_mamba3d_c3vd",
            eval_bridge="src.unified_testing.unified_test",
            checkpoint_path=None,
            baseline_repo_path=None,
            capabilities=AdapterCapabilities(
                supports_train=True,
                supports_eval=True,
                accepts_normals=False,
                accepts_variable_points=False,
                allowed_preprocess_profiles=legacy_profiles,
                private_input_transform_id="bn3_c3vd_unit_cube_shared",
                refinement_mode="none",
            ),
            baseline_normalization_policy_id="c3vd_pointnetlk_style_unit_scale",
            baseline_normalization_reference=(
                "forks/PointNetLK_c3vd/ptlk/data/datasets.py"
            ),
            default_eval_normalize_mode="unit_cube",
        ),
        ModelSpec(
            model_id="mamba3d_mamba2_direct",
            family="learning",
            source_kind="benchmark_native",
            runtime_policy="cuda_required",
            adapter_path=(
                "src.unified_testing.adapters.pointnetlk_c3vd_adapter."
                "PointNetLKC3VDAdapter"
            ),
            train_bridge="src.benchmarking.bridges.train_mamba3d_c3vd",
            eval_bridge="src.unified_testing.unified_test",
            checkpoint_path=None,
            baseline_repo_path=None,
            capabilities=AdapterCapabilities(
                supports_train=True,
                supports_eval=True,
                accepts_normals=False,
                accepts_variable_points=False,
                allowed_preprocess_profiles=legacy_profiles,
                private_input_transform_id="bn3_c3vd_unit_cube_shared",
                refinement_mode="none",
            ),
            baseline_normalization_policy_id="c3vd_pointnetlk_style_unit_scale",
            baseline_normalization_reference=(
                "forks/PointNetLK_c3vd/ptlk/data/datasets.py"
            ),
            default_eval_normalize_mode="unit_cube",
        ),
        ModelSpec(
            model_id="mambanetlk",
            family="learning",
            source_kind="benchmark_native",
            runtime_policy="cuda_required",
            adapter_path=(
                "src.unified_testing.adapters.pointnetlk_c3vd_adapter."
                "PointNetLKC3VDAdapter"
            ),
            train_bridge="src.benchmarking.bridges.train_mamba3d_c3vd",
            eval_bridge="src.unified_testing.unified_test",
            checkpoint_path=None,
            baseline_repo_path=None,
            capabilities=AdapterCapabilities(
                supports_train=True,
                supports_eval=True,
                accepts_normals=False,
                accepts_variable_points=False,
                allowed_preprocess_profiles=legacy_profiles,
                private_input_transform_id="bn3_c3vd_unit_cube_shared",
                refinement_mode="lk",
            ),
            baseline_normalization_policy_id="c3vd_pointnetlk_style_unit_scale",
            baseline_normalization_reference=(
                "forks/PointNetLK_c3vd/ptlk/data/datasets.py"
            ),
            default_eval_normalize_mode="unit_cube",
        ),
        ModelSpec(
            model_id="bufferx",
            family="learning",
            source_kind="vendor_readonly",
            runtime_policy="cuda_required",
            adapter_path="src.unified_testing.adapters.bufferx_adapter.BufferXAdapter",
            train_bridge="src.benchmarking.bridges.train_bufferx_c3vd",
            eval_bridge="src.unified_testing.unified_test",
            checkpoint_path=None,
            baseline_repo_path="baselines/BUFFER-X",
            capabilities=AdapterCapabilities(
                supports_train=True,
                supports_eval=True,
                accepts_normals=False,
                accepts_variable_points=False,
                allowed_preprocess_profiles=main_profiles,
                private_input_transform_id="bufferx_metric_private_downsample",
                refinement_mode="none",
            ),
            baseline_normalization_policy_id="bufferx_metric_scene_no_global_norm",
            baseline_normalization_reference=(
                "baselines/BUFFER-X/config/indoor_config.py"
            ),
            default_eval_normalize_mode="none",
        ),
        ModelSpec(
            model_id="regtr",
            family="learning",
            source_kind="vendor_readonly",
            runtime_policy="cuda_required",
            adapter_path="src.unified_testing.adapters.regtr_adapter.RegTRAdapter",
            train_bridge="src.benchmarking.bridges.train_regtr_c3vd",
            eval_bridge="src.unified_testing.unified_test",
            checkpoint_path=None,
            baseline_repo_path="baselines/RegTR",
            capabilities=AdapterCapabilities(
                supports_train=True,
                supports_eval=True,
                accepts_normals=False,
                accepts_variable_points=True,
                allowed_preprocess_profiles=main_profiles,
                private_input_transform_id="bn3_metric_unit",
                refinement_mode="none",
            ),
            baseline_normalization_policy_id="metric_raw_no_global_norm",
            baseline_normalization_reference="baselines/RegTR/src/data_loaders/threedmatch.py",
            default_eval_normalize_mode="none",
        ),
        ModelSpec(
            model_id="geotransformer",
            family="learning",
            source_kind="vendor_readonly",
            runtime_policy="cuda_required",
            adapter_path=(
                "src.unified_testing.adapters.geotransformer_adapter."
                "GeoTransformerAdapter"
            ),
            train_bridge="src.benchmarking.bridges.train_geotransformer_c3vd",
            eval_bridge="src.unified_testing.unified_test",
            checkpoint_path=None,
            baseline_repo_path="baselines/GeoTransformer",
            capabilities=AdapterCapabilities(
                supports_train=True,
                supports_eval=True,
                accepts_normals=False,
                accepts_variable_points=True,
                allowed_preprocess_profiles=main_profiles,
                private_input_transform_id="bn3_metric_unit",
                refinement_mode="none",
            ),
            baseline_normalization_policy_id="threedmatch_metric_raw",
            baseline_normalization_reference=(
                "baselines/GeoTransformer/geotransformer/datasets/registration/"
                "threedmatch/dataset.py"
            ),
            default_eval_normalize_mode="none",
        ),
    )
