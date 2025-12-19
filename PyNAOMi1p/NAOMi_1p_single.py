"""Python translation of NAOMi_1p_single.m with stubbed dependencies."""

from __future__ import annotations

import copy
import pickle
import time
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

from .config import load_rush_ai148d_config
from .optics import simulate_1p_optical_propagation
from .scanning import scan_volume_1p
from .time_trace import fun_time_trace_generation
from .volume import simulate_neural_volume


def ensure_unique_dir(base: Path) -> Path:
    """Create an incremented folder similar to MATLAB logic."""

    idx = 1
    while True:
        candidate = base / str(idx)
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        idx += 1


def save_pickle(path: Path, data: Any) -> None:
    with open(path, "wb") as handle:
        pickle.dump(data, handle)


def save_numpy(path: Path, array: np.ndarray) -> None:
    np.save(path, array)


def save_heatmap(data: np.ndarray, title: str, path: Path) -> None:
    plt.figure(figsize=(4, 6))
    plt.imshow(data, aspect="auto", cmap="viridis")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def build_output_base(
    base_root: Path,
    mode: str,
    pixel_size: float,
    vol_params: Dict[str, Any],
    psf_params: Dict[str, Any],
    frate: float,
    exp_level: float,
    neur_density: float,
    avg_power: float,
) -> Path:
    vol_sz = vol_params["vol_sz"]
    descriptor = (
        f"vol_{vol_sz[0]}_{vol_sz[2]}_NA_{psf_params['objNA']:.2f}_Hz_{int(frate)}"
        f"_exp_{int(exp_level)}_d_{int(neur_density / 1e3)}k_pw_{avg_power:.2f}"
    )
    return base_root / mode / f"res_{pixel_size:.2f}" / descriptor


def translate_naomi_single(output_root: Path | None = None) -> Dict[str, Any]:
    """Entry point mirroring NAOMi_1p_single.m."""

    cfg = load_rush_ai148d_config()
    vol_params = copy.deepcopy(cfg.vol_params)
    neur_params = copy.deepcopy(cfg.neur_params)
    vasc_params = copy.deepcopy(cfg.vasc_params)
    dend_params = copy.deepcopy(cfg.dend_params)
    bg_params = copy.deepcopy(cfg.bg_params)
    axon_params = copy.deepcopy(cfg.axon_params)
    psf_params = copy.deepcopy(cfg.psf_params)
    scan_params = copy.deepcopy(cfg.scan_params)
    spike_opts = copy.deepcopy(cfg.spike_opts)
    wdm_params = copy.deepcopy(cfg.wdm_params)
    noise_params = copy.deepcopy(cfg.noise_params)

    FOV_sz = 600
    nt = 1000
    fn = 10
    pavg = 0.5

    vol_params["vol_sz"][0] = FOV_sz
    vol_params["vol_sz"][1] = FOV_sz
    spike_opts["nt"] = nt + 200
    spike_opts["dt"] = 1 / fn
    wdm_params["pavg"] = pavg
    spike_opts.setdefault("rate", 1e-3)
    spike_opts.setdefault("smod_flag", "other")

    vessel_mod = {
        "flag": True,
        "frate": fn,
        "vessel_dynamics": 2,
        "FOV_N": 2,
        "max_dilate_amp": 10,
        "seed": 10,
    }

    output_root = output_root or Path("PyNAOMi_output")
    output_root = Path(output_root)
    base_dir = build_output_base(
        output_root,
        cfg.mode,
        cfg.pixel_size,
        vol_params,
        psf_params,
        cfg.frate,
        cfg.exp_level,
        vol_params["neur_density"],
        wdm_params["pavg"],
    )
    output_dir = ensure_unique_dir(base_dir)

    start = time.perf_counter()
    (
        vol_out,
        vol_params,
        neur_params,
        vasc_params,
        dend_params,
        bg_params,
        axon_params,
    ) = simulate_neural_volume(
        vol_params,
        neur_params,
        vasc_params,
        dend_params,
        bg_params,
        axon_params,
        psf_params,
    )
    elapsed = time.perf_counter() - start
    print(f"Simulated neural volume in {elapsed:.2f} seconds.")

    spike_opts["K"] = vol_out.get("gp_vals", np.zeros((1, 1))).shape[0]

    save_pickle(output_dir / "vol_out.pkl", vol_out)
    save_pickle(
        output_dir / "params.pkl",
        {
            "vol_params": vol_params,
            "neur_params": neur_params,
            "vasc_params": vasc_params,
            "dend_params": dend_params,
            "bg_params": bg_params,
            "axon_params": axon_params,
        },
    )
    save_numpy(output_dir / "neur_vol.npy", vol_out["neur_vol"])
    save_numpy(output_dir / "neur_ves_all.npy", vol_out["neur_ves_all"])

    start = time.perf_counter()
    psf_struct = simulate_1p_optical_propagation(vol_params, psf_params, vol_out)
    elapsed = time.perf_counter() - start
    print(f"Simulated optical propagation in {elapsed:.2f} seconds.")
    save_pickle(output_dir / "psf_struct.pkl", psf_struct)

    trace_dir = output_dir / f"firing_rate_{spike_opts['rate']}_smod_flag_{spike_opts['smod_flag']}"
    traces = fun_time_trace_generation(vol_out, nt, fn, trace_dir)

    save_heatmap(traces["soma"], "soma", output_dir / "soma_heat.png")
    save_heatmap(traces["bg"], "bg", output_dir / "bg_heat.png")
    save_heatmap(traces["dend"], "dendrites", output_dir / "dend_heat.png")

    start = time.perf_counter()
    movie = scan_volume_1p(
        vol_out,
        psf_struct,
        traces,
        vol_params,
        scan_params,
        noise_params,
        spike_opts,
        wdm_params,
        vessel_mod,
        cfg.pixel_size,
        cfg.exp_level,
        output_dir,
    )
    elapsed = time.perf_counter() - start
    print(f"Simulated recordings in {elapsed:.2f} seconds.")

    return {
        "output_dir": output_dir,
        "vol_out": vol_out,
        "psf_struct": psf_struct,
        "traces": traces,
        "movie": movie,
    }


if __name__ == "__main__":
    translate_naomi_single()
