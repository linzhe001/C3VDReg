# Agent Proposal: Prompt Chain

## Sources
- `/home/linzhe/PCLR_compare/outputs/benchmark/hparam_transfer/regtr_measured_run/context/context_pack.json`
- `/home/linzhe/PCLR_compare/outputs/benchmark/hparam_transfer/regtr_measured_run/proposal/agent_proposal.yaml`

## pre-LLM prompt package

```text
ROLE: evidence-constrained DPG-HPT proposal author

INPUTS:
- target: c3vd_raycasting_v1 | medical_endoscopic_cross_modal
- unit/pair: mm_like | partial_to_partial
- pose: source_to_target | ['4x4_homogeneous']
- 3DMatch: indoor_metric_scene (baselines/RegTR/src/conf/3dmatch.yaml)
- candidate_limit: 3
- allowed_parameters: 27
- locked_parameters: 12

INSTRUCTION:
Read only the frozen context pack, dataset profiles, model route cards,
public config evidence, and transfer rules. Select/reject routes with
evidence. Fill conservative/default/aggressive candidates only for
allowlisted fields. For every field, write value, owner/status, source
route, conversion rule, and evidence path. Preserve
used_official_test_feedback=false. Abstain rather than invent evidence.
```

## auditable exchange note

```text
The retained reproducible chain is prompt package ->
agent_proposal.yaml -> validator outputs -> candidate configs.
No hidden private reasoning transcript is required or available;
field-level rationale and evidence paths are explicit in YAML.
```
