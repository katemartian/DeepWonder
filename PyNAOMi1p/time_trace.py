"""Placeholder routines for time-trace generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np


@dataclass
class SpikeOptions:
    dt: float
    nt: int
    rate: float


def fun_time_trace_generation(vol_out: Dict[str, Any], nt: int, frate: float, output_dir: Path) -> Dict[str, np.ndarray]:
    """Dummy spike-generation routine.

    Generates random soma/dendrite/background activity arrays and persists
    them as NumPy binaries to mimic MATLAB's saved structures.
    """

    num_cells = vol_out.get("gp_vals", np.zeros((0, 1))).shape[0] or 1
    rng = np.random.default_rng(seed=42)
    soma = rng.random((num_cells, nt), dtype=np.float32)
    dend = rng.random((num_cells, nt), dtype=np.float32)
    bg = rng.random((1, nt), dtype=np.float32)

    traces = {"soma": soma, "dend": dend, "bg": bg}

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "S.npy", traces)

    opts = {"dt": 1 / frate, "nt": nt, "rate": 1e-3}
    np.save(output_dir / "spikes_opts.npy", opts)

    return traces
