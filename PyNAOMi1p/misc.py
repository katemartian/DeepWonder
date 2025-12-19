"""Miscellaneous helper functions translated from MATLAB."""

from __future__ import annotations

from typing import Optional

import numpy as np


def masked_3dgp_v2(
    grid_sz: np.ndarray | list | tuple | int,
    l_scale: np.ndarray | list | tuple | float,
    p_scale: float,
    mu: float | np.ndarray = 0.0,
    bin_mask: np.ndarray | float | int | None = None,
    threshold: float = 1e-10,
    l_weights: np.ndarray | list | tuple | float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Draw from a 3D GP using FFT filtering (translation of masked_3DGP_v2.m)."""

    if bin_mask is None:
        bin_mask = 1.0
    grid_sz = np.asarray(grid_sz, dtype=int).ravel()
    if grid_sz.size == 1:
        grid_sz = np.array([grid_sz[0], grid_sz[0], grid_sz[0]], dtype=int)

    l_scale = np.asarray(l_scale, dtype=float)
    if l_scale.ndim == 0:
        l_scale = np.array([l_scale, l_scale, l_scale], dtype=float)
    if l_scale.ndim == 1:
        if l_scale.size == 1:
            l_scale = np.array([l_scale[0], l_scale[0], l_scale[0]], dtype=float)
        l_scale = l_scale.reshape(1, 3)
    if l_scale.shape[1] == 1:
        l_scale = np.repeat(l_scale, 3, axis=1)

    l_weights = np.asarray(l_weights, dtype=float)
    if l_weights.ndim == 0:
        l_weights = np.ones(l_scale.shape[0], dtype=float) * float(l_weights)
    if l_weights.size == 1 and l_scale.shape[0] > 1:
        l_weights = np.ones(l_scale.shape[0], dtype=float) * float(l_weights)

    if rng is None:
        rng = np.random.default_rng()

    wmx = np.pi / 2.0
    grid_x = (np.linspace(-wmx, wmx, grid_sz[0], dtype=np.float32) ** 2).reshape(-1, 1, 1)
    grid_y = (np.linspace(-wmx, wmx, grid_sz[1], dtype=np.float32) ** 2).reshape(1, -1, 1)
    grid_z = (np.linspace(-wmx, wmx, grid_sz[2], dtype=np.float32) ** 2).reshape(1, 1, -1)

    gp_vals = np.zeros(grid_sz, dtype=np.complex64)
    half_thresh = np.prod(grid_sz) / 2.0
    for i in range(l_scale.shape[0]):
        ker_x = np.exp(-grid_x * (l_scale[i, 0] ** 2))
        ker_y = np.exp(-grid_y * (l_scale[i, 1] ** 2))
        ker_z = np.exp(-grid_z * (l_scale[i, 2] ** 2))
        ker_1 = ker_x * ker_y * ker_z
        ker_loc = ker_1 > threshold
        ker_len = int(ker_loc.sum())
        scale = l_weights[i] * np.sqrt(np.prod(l_scale[i, :]))
        if ker_len < half_thresh:
            tmp = rng.standard_normal(ker_len).astype(np.float32) + 1j * rng.standard_normal(ker_len).astype(np.float32)
            tmp = scale * tmp * ker_1[ker_loc]
            gp_vals[ker_loc] += tmp
        else:
            tmp = rng.standard_normal(grid_sz).astype(np.float32) + 1j * rng.standard_normal(grid_sz).astype(np.float32)
            tmp = scale * tmp * ker_1
            gp_vals += tmp

    gp_vals = np.sqrt(gp_vals.size) * np.real(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(gp_vals))))
    gp_vals = p_scale * (2 ** 4.5 / np.pi ** 1.5) * bin_mask * gp_vals / np.sqrt(len(l_weights)) + mu
    return gp_vals.astype(np.float32)
