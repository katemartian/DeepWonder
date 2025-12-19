"""Scanning simulation placeholders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import tifffile


def scan_volume_1p(
    vol_out: Dict[str, Any],
    psf_struct: Dict[str, Any],
    neur_act: Dict[str, np.ndarray],
    vol_params: Dict[str, Any],
    scan_params: Dict[str, Any],
    noise_params: Dict[str, Any],
    spike_opts: Dict[str, Any],
    wdm_params: Dict[str, Any],
    vessel_mod: Dict[str, Any],
    pixel_size: float,
    exp_level: float,
    output_dir: Path,
) -> np.ndarray:
    """Produce a fake movie volume and save to disk."""

    soma = neur_act.get("soma")
    if soma is None:
        raise ValueError("Missing soma activity")

    num_frames = soma.shape[1]
    vol_shape = vol_out.get("neur_vol", np.zeros((1, 1, 1))).shape
    movie = np.zeros((*vol_shape[:2], num_frames), dtype=np.float32)

    # Fill movie with averaged activity scaled by exposure level
    mean_activity = soma.mean(axis=0)
    for idx in range(num_frames):
        movie[..., idx] = exp_level * mean_activity[idx]

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "movie.npy", movie)
    _save_tiff_stack(output_dir / "movie.tiff", movie)
    return movie


def _save_tiff_stack(path: Path, movie: np.ndarray) -> None:
    """Normalize movie and store as uint16 TIFF stack."""

    data = movie.astype(np.float32).copy()
    data -= data.min()
    max_val = data.max()
    if max_val > 0:
        data /= max_val
    scaled = np.clip(data * 65535.0, 0, 65535).astype(np.uint16)
    tifffile.imwrite(path, scaled, photometric="minisblack")
