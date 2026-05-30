"""Registry for official benchmark-visible preprocess profiles."""

from __future__ import annotations

from src.benchmarking.preprocess.profiles import DEFAULT_PROFILES, PreprocessProfile


class PreprocessRegistry:
    """In-memory registry of official preprocess profiles."""

    def __init__(self, register_defaults: bool = True) -> None:
        self._profiles: dict[str, PreprocessProfile] = {}
        if register_defaults:
            for profile in DEFAULT_PROFILES:
                self.register(profile)

    def register(self, profile: PreprocessProfile) -> None:
        self._profiles[profile.profile_id] = profile

    def get(self, profile_id: str) -> PreprocessProfile:
        try:
            return self._profiles[profile_id]
        except KeyError as exc:
            raise KeyError(f"Unknown preprocess profile '{profile_id}'.") from exc

    def list_profiles(self) -> tuple[str, ...]:
        return tuple(sorted(self._profiles))
