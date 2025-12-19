"""Minimal smoke test for volume simulation."""

from __future__ import annotations

import numpy as np

from volume import simulate_neural_volume


def _assert_nonzero(name: str, arr: np.ndarray) -> None:
    if not np.any(arr):
        raise AssertionError(f"{name} is all zeros")


def main() -> None:
    np.random.seed(123)

    vol_params = {
        "vol_sz": [40, 40, 20],
        "min_dist": 12,
        "vres": 2,
        "N_neur": 2,
        "N_den": 1,
        "N_bg": 2,
        "vol_depth": 10,
        "verbose": 0,
        "seed": 1,
    }
    neur_params = {
        "n_samps": 60,
        "seed": 2,
    }
    vasc_params = {"flag": 0}
    dend_params = {
        "dtParams": [3, 8, 6, 1, 1],
        "atParams": [1, 6, 4, 1, 1],
        "atParams2": [1, 6, 4, 1, 1],
        "dims": [10, 10, 10],
        "dimsSS": [2, 2, 2],
        "seed": 3,
    }
    bg_params = {"flag": 0}
    axon_params = {"flag": 0}
    psf_params = {"objNA": 0.8, "n": 1.33}

    vol_out, *_ = simulate_neural_volume(
        vol_params,
        neur_params,
        vasc_params,
        dend_params,
        bg_params,
        axon_params,
        psf_params,
        debug_opt=True,
    )

    vres = vol_params["vres"]
    expected_shape = tuple(np.array(vol_params["vol_sz"]) * vres)

    neur_vol = vol_out["neur_vol"]
    if neur_vol.shape != expected_shape:
        raise AssertionError(f"neur_vol shape {neur_vol.shape} != {expected_shape}")

    neur_soma = vol_out["neur_soma"]
    neur_num = vol_out["neur_num"]
    neur_num_ad = vol_out["neur_num_AD"]
    gp_nuc = vol_out["gp_nuc"]
    gp_soma = vol_out["gp_soma"]

    _assert_nonzero("neur_soma", neur_soma)
    _assert_nonzero("neur_num", neur_num)
    _assert_nonzero("neur_num_AD", neur_num_ad)

    if len(gp_nuc) != vol_params["N_neur"]:
        raise AssertionError("gp_nuc length mismatch")
    if len(gp_soma) != vol_params["N_neur"]:
        raise AssertionError("gp_soma length mismatch")

    tri = vol_out["Tri"]
    if tri is not None and tri.size == 0:
        print("Warning: Tri is empty (SciPy may be unavailable).")

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
