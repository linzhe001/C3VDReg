"""Dataset-profile-guided hyperparameter transfer helpers."""

from src.benchmarking.hparam_transfer.candidate_validation import (
    load_candidate_bundle,
    validate_candidate_configs,
)
from src.benchmarking.hparam_transfer.candidate_validation import (
    render_validation_summary as render_candidate_validation_summary,
)
from src.benchmarking.hparam_transfer.candidate_validation import (
    write_validation_summary as write_candidate_validation_summary,
)
from src.benchmarking.hparam_transfer.context_pack import (
    build_context_pack,
    render_context_pack_markdown,
    write_context_pack_outputs,
)
from src.benchmarking.hparam_transfer.dataset_profiles import (
    measure_dataset_profile,
    render_dataset_profile_markdown,
    write_dataset_profile_outputs,
)
from src.benchmarking.hparam_transfer.promotion import (
    build_promoted_model_config,
    load_base_model_config,
    load_candidate_bundle_payload,
    load_validation_summary,
    write_promoted_artifacts,
)
from src.benchmarking.hparam_transfer.proposal_template import (
    build_agent_proposal_template,
    write_agent_proposal_template,
)
from src.benchmarking.hparam_transfer.proposal_validation import (
    build_transfer_trace,
    load_agent_proposal,
    load_context_pack,
    normalize_candidate_configs,
    validate_proposal,
    write_validation_outputs,
)
from src.benchmarking.hparam_transfer.reference_profiles import (
    export_reference_profiles,
    render_reference_profiles_markdown,
    write_reference_profile_outputs,
)
from src.benchmarking.hparam_transfer.report_rendering import (
    load_validation_payload,
    render_transfer_report,
    render_validation_summary,
    write_transfer_report,
)

__all__ = [
    "build_context_pack",
    "build_agent_proposal_template",
    "build_transfer_trace",
    "build_promoted_model_config",
    "export_reference_profiles",
    "load_candidate_bundle",
    "load_agent_proposal",
    "load_base_model_config",
    "load_candidate_bundle_payload",
    "load_context_pack",
    "load_validation_payload",
    "load_validation_summary",
    "render_context_pack_markdown",
    "render_candidate_validation_summary",
    "write_context_pack_outputs",
    "measure_dataset_profile",
    "normalize_candidate_configs",
    "validate_candidate_configs",
    "render_dataset_profile_markdown",
    "render_reference_profiles_markdown",
    "render_transfer_report",
    "render_validation_summary",
    "validate_proposal",
    "write_agent_proposal_template",
    "write_candidate_validation_summary",
    "write_dataset_profile_outputs",
    "write_promoted_artifacts",
    "write_reference_profile_outputs",
    "write_transfer_report",
    "write_validation_outputs",
]
