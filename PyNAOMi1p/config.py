"""Utility helpers for loading configuration parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Config:
    vol_params: Dict[str, Any]
    neur_params: Dict[str, Any]
    vasc_params: Dict[str, Any]
    dend_params: Dict[str, Any]
    bg_params: Dict[str, Any]
    axon_params: Dict[str, Any]
    psf_params: Dict[str, Any]
    scan_params: Dict[str, Any]
    spike_opts: Dict[str, Any]
    wdm_params: Dict[str, Any]
    noise_params: Dict[str, Any]
    pixel_size: float
    exp_level: float
    mode: str
    frate: float


def load_rush_ai148d_config() -> Config:
    """Return a Python analogue of RUSH_ai148d_config.m settings."""

    vol_params = {
        "vol_sz": [600, 600, 150], 
        "vol_depth": 50, 
        "neur_density": 3e4, 
        "vres": 1
    }
    neur_params = {"avg_rad": 5.9}
    vasc_params: Dict[str, Any] = {}
    dend_params: Dict[str, Any] = {}
    bg_params: Dict[str, Any] = {}
    axon_params: Dict[str, Any] = {}

    FN = 18
    M = 10
    obj_immersion = "air"
    psf_params: Dict[str, Any] = {
        "obj_fl": FN / M,
        "objNA": 0.35,
        "NA": 0.35,
        "lambda": 0.488,
        "psf_sz": [36, 36, 100],
    }
    if obj_immersion == "water":
        psf_params["zernikeWt"] = [0] * 11
        nidx = 1.33
    elif obj_immersion == "air":
        psf_params["zernikeWt"] = [0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0]
        nidx = 1.0
    else:
        raise ValueError("Require water or air as immersion medium")

    scan_params = {"motion": False, "verbose": 2}
    frate = 10
    spike_opts = {
        "prot": "GCaMP6",
        "dendflag": 1,
        "nt": 1100,
        "dt": 1 / frate,
        "rate": 1e-3,
        "smod_flag": "Ca_DE",
        "p_off": -1,
    }

    wdm_params = {"lambda": 0.532, "pavg": 1.0, "qe": 0.7, "nidx": nidx}
    noise_params: Dict[str, Any] = {}

    return Config(
        vol_params=vol_params,
        neur_params=neur_params,
        vasc_params=vasc_params,
        dend_params=dend_params,
        bg_params=bg_params,
        axon_params=axon_params,
        psf_params=psf_params,
        scan_params=scan_params,
        spike_opts=spike_opts,
        wdm_params=wdm_params,
        noise_params=noise_params,
        pixel_size=0.8,
        exp_level=5,
        mode="w_dend",
        frate=10,
    )
