"""Placeholder optical propagation routines."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def gaussian_beam_size(psf_params: Dict[str, Any], dist: float, apod: float = 2.0) -> np.ndarray:
    """Translate gaussianBeamSize.m (overestimate beam waist extent)."""

    na = float(psf_params.get("objNA", psf_params.get("NA", 0.8)))
    refr = float(psf_params.get("n", 1.33))
    return np.ceil(np.tan(np.arcsin(na / refr)) * dist * 1.5) * apod * np.array([1.0, 1.0, 0.0])


def simulate_1p_optical_propagation(
    vol_params: Dict[str, Any],
    psf_params: Dict[str, Any],
    vol_out: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """Generate dummy PSF and masks."""

    shape = vol_out.get("neur_vol", np.zeros((1, 1, 1))).shape
    rng = np.random.default_rng(seed=123)
    psf = rng.random(shape, dtype=np.float32)
    mask = rng.random(shape, dtype=np.float32)
    colmask = rng.random(shape, dtype=np.float32)

    return {"psf": psf, "mask": mask, "colmask": colmask, "psfB": {"mask": mask}, "psfT": {"mask": mask}}
