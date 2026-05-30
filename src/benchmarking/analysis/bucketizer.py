"""Bucket construction helpers for analysis views."""

from __future__ import annotations

SUPPORTED_BUCKET_KEYS = {
    "overlap_bin",
    "rotation_bin",
    "translation_bin",
    "scene_id",
    "preprocess_profile_id",
    "refinement_track",
}


def build_bucket_views(
    records: list[dict[str, object]],
    bucket_keys: list[str],
) -> dict[str, list[dict[str, object]]]:
    """Group records by benchmark bucket keys while preserving total counts."""

    views: dict[str, list[dict[str, object]]] = {}
    for key in bucket_keys:
        if key not in SUPPORTED_BUCKET_KEYS:
            raise ValueError(f"Unsupported bucket key '{key}'.")
        grouped: dict[str, list[dict[str, object]]] = {}
        for record in records:
            bucket_value = record.get(key)
            label = "missing" if bucket_value in {None, ""} else str(bucket_value)
            grouped.setdefault(label, []).append(record)
        views[key] = [
            {
                "bucket_key": key,
                "bucket_value": value,
                "count": len(items),
                "records": items,
            }
            for value, items in sorted(grouped.items())
        ]
    return views
