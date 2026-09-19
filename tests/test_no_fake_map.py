from __future__ import annotations

import pytest

from src.data.download_data import fetch_real_map_incidence


def test_removed_map_fetcher_raises():
    with pytest.raises(RuntimeError, match="simulated"):
        fetch_real_map_incidence()
