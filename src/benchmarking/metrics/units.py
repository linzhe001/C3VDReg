"""Unit conversion helpers for benchmark point coordinates."""

from __future__ import annotations

from typing import Literal

PointUnit = Literal["m", "mm", "mm_like"]

_POINT_UNIT_ALIASES: dict[str, PointUnit] = {
    "m": "m",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "mm": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "millimetre": "mm",
    "millimetres": "mm",
    "mm_like": "mm_like",
    "mm-like": "mm_like",
    "millimeter_like": "mm_like",
    "millimeter-like": "mm_like",
}


def canonicalize_point_unit(point_unit: str) -> PointUnit:
    """Normalize supported point-coordinate unit names."""

    unit = str(point_unit).strip().lower()
    try:
        return _POINT_UNIT_ALIASES[unit]
    except KeyError as exc:
        supported = ", ".join(sorted(_POINT_UNIT_ALIASES))
        raise ValueError(
            f"Unsupported point_unit '{point_unit}'. Supported values: {supported}."
        ) from exc


def point_unit_to_mm_scale(point_unit: str) -> float:
    """Return the multiplier from raw coordinate distance to millimeters."""

    unit = canonicalize_point_unit(point_unit)
    if unit == "m":
        return 1000.0
    return 1.0


def point_unit_to_m_scale(point_unit: str) -> float:
    """Return the multiplier from raw coordinate distance to meters."""

    unit = canonicalize_point_unit(point_unit)
    if unit == "m":
        return 1.0
    return 0.001


def distance_to_millimeters(distance: float, point_unit: str) -> float:
    return float(distance) * point_unit_to_mm_scale(point_unit)


def distance_to_meters(distance: float, point_unit: str) -> float:
    return float(distance) * point_unit_to_m_scale(point_unit)
