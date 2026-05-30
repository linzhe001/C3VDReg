"""
Algorithm adapters for unified testing pipeline.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

POINTNETLK_C3VD_IMPORT_ERROR = None

_ADAPTER_MODULES = {
    "PointNetLKC3VDAdapter": ".pointnetlk_c3vd_adapter",
    "PointNetLKAdapter": ".pointnetlk_adapter",
    "PointNetLKRevisitedAdapter": ".pointnetlk_revisited_adapter",
    "BufferXAdapter": ".bufferx_adapter",
    "RegTRAdapter": ".regtr_adapter",
    "GeoTransformerAdapter": ".geotransformer_adapter",
    "DCPAdapter": ".dcp_adapter",
    "ICPAdapter": ".icp_adapter",
}

__all__ = [
    "POINTNETLK_C3VD_IMPORT_ERROR",
    "PointNetLKC3VDAdapter",
    "PointNetLKAdapter",
    "PointNetLKRevisitedAdapter",
    "BufferXAdapter",
    "RegTRAdapter",
    "GeoTransformerAdapter",
    "DCPAdapter",
    "ICPAdapter",
]


def _load_adapter(name: str) -> Any:
    module = import_module(_ADAPTER_MODULES[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __getattr__(name: str) -> Any:
    global POINTNETLK_C3VD_IMPORT_ERROR

    if name == "POINTNETLK_C3VD_IMPORT_ERROR":
        return POINTNETLK_C3VD_IMPORT_ERROR

    if name == "PointNetLKC3VDAdapter":
        try:
            return _load_adapter(name)
        except Exception as exc:  # pragma: no cover - optional dependency gate
            POINTNETLK_C3VD_IMPORT_ERROR = exc
            globals()[name] = None
            return None

    if name in _ADAPTER_MODULES:
        return _load_adapter(name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
