"""Compatibility layer for the project data-download API.

The repository now works from real incidence and climate sources. Legacy
simulated-data fetchers are intentionally not kept as active code paths.
"""

from __future__ import annotations


def fetch_real_map_incidence() -> None:
    """Raise an explicit error when legacy simulated-map code is requested."""
    raise RuntimeError(
        "This repository now uses real incidence data inputs; the simulated map fetcher is no longer available."
    )


__all__ = ["fetch_real_map_incidence"]
