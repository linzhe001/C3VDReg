"""Rule-based failure tagging and case mining."""

from __future__ import annotations


def assign_failure_tags(record: dict[str, object]) -> list[str]:
    tags: set[str] = set()
    if record.get("overlap_bin") in {"very_low", "low"}:
        tags.add("low_overlap")
    if float(record.get("rre_deg") or 0.0) > 10.0:
        tags.add("large_rotation")
    if float(record.get("rte_mm") or 0.0) > 10.0:
        tags.add("large_translation")
    if float(record.get("visible_nn_p90_mm") or 0.0) > 20.0:
        tags.add("density_mismatch")
    if record.get("artifact_bin") in {"specular", "noisy_depth"}:
        tags.add("specular_or_noisy_depth")
    if (
        record.get("refinement_track") not in {None, "none"}
        and float(record.get("rre_deg") or 0.0) > 5.0
    ):
        tags.add("refinement_divergence")
    if (
        float(record.get("rre_deg") or 0.0) > 60.0
        and float(record.get("rte_mm") or 0.0) < 5.0
    ):
        tags.add("possible_pose_convention_issue")
    return sorted(tags)


def mine_failure_cases(
    records: list[dict[str, object]],
    topk: int,
) -> dict[str, list[dict[str, object]]]:
    by_scene: dict[str, dict[str, object]] = {}
    for record in records:
        scene_id = str(record.get("scene_id", "unknown"))
        current = by_scene.get(scene_id)
        score = (
            float(record.get("rre_deg") or 0.0),
            float(record.get("rte_mm") or 0.0),
        )
        if current is None:
            by_scene[scene_id] = record
            continue
        current_score = (
            float(current.get("rre_deg") or 0.0),
            float(current.get("rte_mm") or 0.0),
        )
        if score > current_score:
            by_scene[scene_id] = record

    return {
        "top_k_worst_rre": sorted(
            records, key=lambda item: float(item.get("rre_deg") or 0.0), reverse=True
        )[:topk],
        "top_k_worst_rte": sorted(
            records, key=lambda item: float(item.get("rte_mm") or 0.0), reverse=True
        )[:topk],
        "top_k_worst_visible_distance": sorted(
            records,
            key=lambda item: float(item.get("visible_nn_mean_mm") or 0.0),
            reverse=True,
        )[:topk],
        "per_scene_worst_case": list(by_scene.values()),
        "failure_gallery_manifest": sorted(
            records,
            key=lambda item: (
                len(item.get("failure_tags", [])),
                float(item.get("rre_deg") or 0.0),
                float(item.get("rte_mm") or 0.0),
            ),
            reverse=True,
        )[:topk],
    }
