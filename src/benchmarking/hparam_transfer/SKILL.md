# Hparam Transfer

## Purpose

This file is the stable repository-local guidance for `DPG-HPT` execution.
Use it when converting a baseline into the benchmark-facing hyperparameter-transfer flow, or when producing:

- `reference_profiles`
- `context_pack`
- `agent_proposal`
- `proposal_validation`
- `candidate_configs`
- `transfer_report`
- `candidate_validation`
- `promotion`

## Required References

Read these before changing the stable protocol:

- `PROJECT_STATE.json`
- `project_map.json`
- `docs/DPG_HPT_Methodology.md`
- `docs/Dataset_Profile_Guided_Hparam_Transfer_Plan.md`
- `docs/Benchmark_Protocol_And_Hparam_Transfer_Summary.md`
- `docs/Benchmark_Efficiency_Table_Design.md`

## Core Rules

1. Treat the protocol as `evidence-constrained configuration transfer`, not free-form tuning.
2. Keep architecture, optimizer family, benchmark metrics, and locked groups fixed unless the user explicitly changes the protocol.
3. Tie transferable parameters to concrete vendor config/code evidence.
4. Use measured target profiles when available; only fall back to durable profile stubs when the run is explicitly marked as fallback.
5. Treat unit semantics as a hard gate. Do not accept `declared_point_unit` blindly when raw geometry scale contradicts it.
6. Treat pose transform format, shape, direction, and translation unit as a hard gate. A profile is incomplete if it records geometry/unit statistics but omits the source pose field and the source-to-target vs target-to-source convention.
7. Build or refresh the common registration dataset census before model-specific transfer. The census must include C3VD raycasting and the common point-cloud registration datasets relevant to the baseline family.
8. Detect how the target baseline actually uses those common datasets before proposing transfers. The model route includes dataset family, domain, pair type, coordinate unit, normalization mode, private model normalization, point budget, voxel/radius scales, pose shape/direction, and evaluation radius.
9. Transfer candidates must be justified by both dataset-profile compatibility and model-specific route evidence. Do not hard-code a single source dataset as the transfer route.
10. Give 3DMatch/3DLoMatch special attention because many registration baselines expose metric-scene configs there, but use them as high-priority evidence within the dataset census, not as an unconditional default.
11. A C3VD train candidate is valid only after dataset-profile comparison, model-route profile generation, cross-profile compatibility analysis, and field-level train-hparam decisions are all recorded.

## Main-Table Point Budget

The primary main-table raw point budget is a benchmark-owned policy, not a free transfer knob.

- Default primary main-table budget is `8192` raw points.
- `12288` is only an optional high-resolution table, not the default main-table setting.
- Do not reinterpret vendor `point_limit`, `num_points`, or similar density/count-like settings as permission to change the benchmark-wide main-table budget.
- For main-table proposals, record `data.num_points = 8192` as `benchmark_owned` unless the user explicitly freezes a different benchmark-wide budget.

## Unit and Perturbation Gate

Before proposing, training, evaluating, or repairing a result bundle:

- Verify `benchmark.point_unit`, manifest/profile unit, raw PLY coordinate scale, and metric conversion agree.
- Verify the target profile `pose` section before using a candidate: source storage field, matrix shape, compact vs homogeneous representation, pose direction, translation unit, and SE(3) validity must all be recorded.
- For C3VD, manifest `gt_transform` is the canonical benchmark pose field: it is a 4x4 homogeneous transform in raw coordinate units and follows the benchmark source-to-target contract used by metrics.
- Baseline adapters must explicitly convert from the benchmark contract when vendor code expects a different pose shape or direction. For example, RegTR training consumes compact 3x4 `pose` tensors and must receive source-to-target transforms after perturbation handling.
- For C3VD, current raw coordinates are `mm_like`: raw distance `x` should report as `x mm`, not `x * 1000 mm`.
- Check physical plausibility with raw PLY bbox extent and pose translation scale. Do not choose a unit because it improves the score.
- Record the evidence if a post-hoc unit repair is needed: scaled fields, scale factor, recomputed fields, backup location, and artifacts that were not regenerated.
- Treat `translation_m` with care. The current perturbation implementation applies it directly in raw coordinate units. On `mm_like` C3VD, `translation_m=0.5` means about `0.5mm`, not `0.5m`.
- Train/eval configs must align on `point_unit`, `preprocess.num_points_override`, `normalize_mode`, and perturbation. Any mismatch must be reported as a distribution shift.

## Dataset Census, Model Route, and Normalization Gate

Before writing a `context_pack`, `agent_proposal`, `candidate_configs`, or `transfer_report`:

1. Build the dataset census.

   - Inspect `src/benchmarking/hparam_transfer/dataset_profiles/dataset_profiles.yaml` and materialized profiles under `src/benchmarking/hparam_transfer/dataset_profiles/`.
   - Include C3VD raycasting and every common point-cloud registration dataset that is already represented in the repo or mentioned by the target baseline's vendor configs/code. At minimum, check the existing registry entries such as `3DMatch`, `3DLoMatch`, `ModelNet40`, `KITTI`, and `ScanNet++_iPhone`; add evidence-backed stubs for other common datasets when the model uses them.
   - For each dataset, record domain, source/target modality, pair type, coordinate unit, inferred unit, route hint, normalization route, raw/filtered point count, bbox diagonal, nearest-neighbor spacing, density/overlap hints when available, split policy, pose storage field, pose transform shape, pose direction, pose translation unit, perturbation unit, evaluation radius, and evidence source.
   - Profile generation must inspect actual pose records, not only manifest metadata. For manifest-backed profiles, summarize transform shape counts, explicit-vs-defaulted transform counts, translation norm percentiles, rotation determinant/orthonormality checks, valid SE(3) fraction, and the code evidence that establishes direction semantics.
   - Do not fabricate missing statistics. Mark unavailable fields as missing and keep the evidence status visible.

2. Audit the target model's dataset routes.

   - Inspect `configs/benchmark/hparam_transfer/baseline_routes.yaml` for the baseline's known routes and preferred/rejected C3VD references.
   - Inspect every route source listed there: vendor config/code when available, or the structured reference stub under `configs/benchmark/hparam_transfer/reference_configs/`.
   - Search the baseline for additional dataset configs if the route card is incomplete. Add the discovered route to the proposal/report evidence before using it.
   - For each model route, record the source dataset, route type, normalization, point budget, pose shape/direction convention, voxel/radius scales, patch/keypoint limits, acceptance radius, data augmentation, and the exact config/code paths that define them.
   - Separate vendor-visible preprocessing from adapter-private transforms and benchmark-owned preprocessing. For example, record `preprocess.normalize_mode` separately from `model.private_normalization_route`.

3. Transfer to C3VD by combining both sources of evidence.

   - First compare C3VD raycasting against the dataset census to identify compatible reference families by domain, modality, pair type, unit/scale, density, normalization route, and evaluation radius.
   - Then intersect those compatible datasets with the target model's actual routes. A route cannot drive transfer if the baseline never used or exposed it.
   - Prefer model-supported metric/local scene routes when C3VD compatibility supports them. `3DMatch`/`3DLoMatch` often deserve priority here, but they still need explicit model-route evidence.
   - Treat `ModelNet40` `unit_cube`/normalized-object routes and `KITTI`/outdoor-lidar routes as mismatched for C3VD metric/endoscopic scene transfer unless a specific parameter is only density/runtime related or the mismatch is explicitly reported.
   - Transfer only fields whose semantics are supported by the route evidence and allowlists, such as voxel size, matching radius, patch radius, point limit, `spatial_scale`, and acceptance radius.
   - Keep the benchmark-owned main-table raw point budget at `8192` unless the user explicitly changes the benchmark policy.

4. Apply the normalization gate.

   - For 3DMatch-like metric-scene routes, preserve metric-space assumptions by default: `normalize_mode: none` unless the vendor code explicitly applies another normalization.
   - For ModelNet-like routes, treat `unit_cube`/normalized-object assumptions as a different operating route, not as a silent default for C3VD.
   - Never infer normalization from score alone. A candidate must cite dataset-census compatibility and model-route evidence before `normalize_mode`, point budget, or radius-like fields are accepted.
   - Train and eval must use the same effective normalization route. If train uses a bridge-specific `unit_cube` path while eval uses raw metric space, report the mismatch as invalid unless explicitly approved for an ablation.

## Cross-Profile Compatibility and C3VD Train Hparams

Before accepting a C3VD training candidate, the `context_pack`, `agent_proposal`, or `transfer_report` must contain a cross-profile comparison that makes the route decision auditable.

The comparison must include:

- C3VD target profile fields: domain, source/target modality, pair type, coordinate unit, inferred unit, normalization route, point-count profile, bbox/spacing profile, split policy, pose storage/shape/direction/translation unit, perturbation unit, and evaluation/reporting unit.
- Target-model route profiles from `reference_profiles`: every dataset route the model actually exposes, including source dataset, route type, normalization, point budget, pose shape/direction convention, scale/radius fields, patch/keypoint limits, data augmentation, acceptance radius, config path, and confidence.
- Compatibility status for each model-supported route against C3VD: `preferred`, `usable_with_risk`, `density_only`, `runtime_only`, or `rejected`.
- Rejection reasons for every unsupported or mismatched route, especially object-vs-scene, metric-vs-normalized, indoor-vs-outdoor, overlap regime, unit scale, and missing vendor evidence.

C3VD train-hparam decisions must then be derived from that comparison:

- Start from model-supported routes that are profile-compatible with C3VD. A dataset may be similar to C3VD, but it cannot drive transfer if the target model never exposes that route.
- For each changed field, record `parameter_path`, chosen value, owner (`benchmark_owned`, `transferred`, `adapter_private`, `runtime_only`, or `locked`), source route/profile, conversion rule, and evidence path.
- For main-table C3VD training, keep raw point budget at `8192` and record it as benchmark-owned unless the user explicitly changes the benchmark-wide policy.
- Preserve C3VD `point_unit=mm_like` and require train/eval agreement on `preprocess.num_points_override`, `normalize_mode`, private normalization route, perturbation, and reporting metric units.
- Preserve C3VD pose semantics: benchmark `gt_transform` is 4x4 source-to-target, and any baseline-private compact pose tensor or inverse transform must be documented as an adapter conversion rather than a dataset-profile change.
- Transfer length-like parameters only when unit and scale compatibility are explicit. If the source route is `unit_cube`/ModelNet-like, do not transfer metric voxel/radius/acceptance thresholds into raw C3VD space without an explicit conversion and risk note.
- Treat runtime settings such as batch size, workers, memory-safe point budget, and checkpoint resume separately from scientific hyperparameters.
- Keep architecture capacity, optimizer family, loss family, and official metric thresholds locked unless the user explicitly approves a protocol change.
- Do not mark a candidate as C3VD train-ready if any unit, normalization, train/eval protocol, or route-evidence mismatch remains unresolved.

## Dataset Profile and Analysis Locations

Use these locations when answering "where is the dataset analysis?":

- The dataset census is materialized by the durable registry plus dataset profile artifacts under `src/benchmarking/hparam_transfer/dataset_profiles/`. In the current implementation it is not a separate `dataset_census.json` file.
- Durable profile registry and reference stubs:
  - `src/benchmarking/hparam_transfer/dataset_profiles/dataset_profiles.yaml`
  - `src/benchmarking/hparam_transfer/dataset_profiles/*.json`
  - `src/benchmarking/hparam_transfer/dataset_profiles/*.md`
  - `configs/benchmark/hparam_transfer/baseline_routes.yaml`
  - `configs/benchmark/hparam_transfer/reference_configs/`
- Measured target dataset profile exports:
  - `src/benchmarking/hparam_transfer/dataset_profiles/c3vd_raycasting_v1.json`
  - `src/benchmarking/hparam_transfer/dataset_profiles/c3vd_raycasting_v1.md`
- Model-specific reference profile exports:
  - `outputs/benchmark/hparam_transfer/<baseline>_measured_run/reference_profiles/*_reference_profiles.json`
  - `outputs/benchmark/hparam_transfer/<baseline>_measured_run/reference_profiles/*_reference_profiles.md`
- Per-baseline transfer analysis:
  - `outputs/benchmark/hparam_transfer/<baseline>_measured_run/context/context_pack.json`
  - `outputs/benchmark/hparam_transfer/<baseline>_measured_run/context/context_pack.md`
  - `outputs/benchmark/hparam_transfer/<baseline>_measured_run/report/transfer_report.md`
  - `outputs/benchmark/hparam_transfer/<baseline>_measured_run/validated/transfer_trace.json`
- Dataset profile implementation:
  - `src/benchmarking/hparam_transfer/dataset_profiles.py`
  - `scripts/benchmark/export_dataset_profile.py`
  - `scripts/benchmark/export_baseline_reference_profiles.py`
- General benchmark result analysis is separate from DPG-HPT dataset profiles:
  - `configs/benchmark/analysis/default.yaml`
  - `src/benchmarking/analysis/`
  - per-run `summary_overview.*`, `run_card.json`, and diagnostics under benchmark eval output directories.

## Recommended Workflow and Artifact Order

1. Review or refresh the dataset census in `src/benchmarking/hparam_transfer/dataset_profiles/`.
2. Export `reference_profiles`; this is the model-specific route audit artifact.
3. Build `context_pack`; it combines the C3VD target profile, model route audit, route card, transfer rules, and policy gates.
4. Record the cross-profile compatibility analysis and selected/rejected routes.
5. Write `agent_proposal` with field-level C3VD train-hparam decisions.
6. Run `proposal_validation`.
7. Render `transfer_report`.
8. Run `candidate_validation`.
9. Request explicit approval before `promotion`.

## Training and Checkpoint Traceability

Before full training after training-related code changes, follow the repository
training rule: create a semantic commit first.

For any trained candidate, keep:

- normalized train config
- generated bridge config
- `git_snapshot.json`
- `run_metadata.json`
- train metrics
- checkpoint used by eval
- best-epoch checkpoint when it differs from the eval checkpoint or is needed to audit selection

For GeoTransformer specifically:

- `model.spatial_scale` and `model.num_points_in_patch` are the primary C3VD scale-sensitive fields.
- `model.neighbor_limits` is an adapter/runtime knob; keep it benchmark-owned unless there is explicit evidence and user approval.
- Verify that `geotransformer_c3vd_model_best.pth` is selected by validation metric. If a historical run overwrote best with the last epoch, record that limitation or rerun with fixed best-checkpoint logic.

## Canonical Scripts

Prefer these entrypoints:

- `scripts/benchmark/export_dataset_profile.py`
- `scripts/benchmark/export_baseline_reference_profiles.py`
- `scripts/benchmark/prepare_hparam_transfer_context.py`
- `scripts/benchmark/init_hparam_transfer_proposal.py`
- `scripts/benchmark/validate_hparam_proposal.py`
- `scripts/benchmark/render_hparam_transfer_report.py`
- `scripts/benchmark/validate_hparam_candidates.py`
- `scripts/benchmark/promote_hparam_candidate.py`

## Output Rules

- Keep natural-language reports in the user's language.
- Keep YAML/JSON keys, file names, and parameter paths in English.
- Mark stub-based target profiles explicitly as a risk.
- Distinguish clearly between:
  - vendor-exposed evidence fields
  - benchmark-owned knobs
  - adapter-only runtime knobs
  - locked parameters
- Mark dry-run candidate validation explicitly when `execute_eval=false`.
- Do not use official test feedback for candidate selection. If a validation score is used for selection, label the candidate as `val_selected`.

## Output Hygiene

Keep `outputs/benchmark` small enough to audit:

- Retain the current manifest/configs, DPG-HPT context/proposal/validation/report, repaired/final eval bundle, run metadata, eval checkpoint, and necessary best-epoch checkpoint.
- Remove or cold-archive old `archive/`, failed/intermediate train runs, obsolete eval bundles, repeated epoch checkpoints, and temporary repair backups once their conclusions are summarized.
- Update `outputs/benchmark/README.md` after cleanup so retained artifacts and deleted categories are explicit.

## Maintenance

- If stable protocol files change, sync `project_map.json`.
- If a baseline-specific transfer rule becomes benchmark policy, encode it in the stable DPG-HPT assets rather than leaving it only in ad hoc run artifacts.
