"""Python translation of NAOMi VolumeCode utilities (simplified)."""

from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path
import time
from typing import Any, Dict, Tuple

import numpy as np

from misc import masked_3dgp_v2
from optics import gaussian_beam_size

def _vec_linspace(v1: np.ndarray, v2: np.ndarray, n: int) -> np.ndarray:
    v1 = np.asarray(v1, dtype=float).reshape(-1, 1)
    v2 = np.asarray(v2, dtype=float).reshape(-1, 1)
    steps = np.linspace(0.0, 1.0, max(1, n))
    return (v1 + (v2 - v1) * steps).astype(float)


def _disk_structuring_element(radius: int) -> np.ndarray:
    rad = max(1, int(np.ceil(radius)))
    y, x = np.ogrid[-rad : rad + 1, -rad : rad + 1]
    return x * x + y * y <= radius * radius


def _binary_dilate(mask: np.ndarray, structure: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    result = mask.copy()
    offsets = np.argwhere(structure)
    center = np.array(structure.shape) // 2
    for offset in offsets:
        dy, dx = (offset - center).astype(int)
        shifted = np.zeros_like(mask, dtype=bool)
        y_start = max(0, dy)
        y_end = mask.shape[0] + min(0, dy)
        x_start = max(0, dx)
        x_end = mask.shape[1] + min(0, dx)
        src_y_start = max(0, -dy)
        src_y_end = src_y_start + (y_end - y_start)
        src_x_start = max(0, -dx)
        src_x_end = src_x_start + (x_end - x_start)
        shifted[y_start:y_end, x_start:x_end] = mask[src_y_start:src_y_end, src_x_start:src_x_end]
        result |= shifted
    return result


def _ball_structuring_element(radius: int) -> np.ndarray:
    rad = max(1, int(np.ceil(radius)))
    z, y, x = np.ogrid[-rad : rad + 1, -rad : rad + 1, -rad : rad + 1]
    return x * x + y * y + z * z <= radius * radius


def _binary_dilate_3d(mask: np.ndarray, structure: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    result = mask.copy()
    offsets = np.argwhere(structure)
    center = np.array(structure.shape) // 2
    for offset in offsets:
        dz, dy, dx = (offset - center).astype(int)
        shifted = np.zeros_like(mask, dtype=bool)
        z_start = max(0, dz)
        z_end = mask.shape[0] + min(0, dz)
        y_start = max(0, dy)
        y_end = mask.shape[1] + min(0, dy)
        x_start = max(0, dx)
        x_end = mask.shape[2] + min(0, dx)
        src_z_start = max(0, -dz)
        src_z_end = src_z_start + (z_end - z_start)
        src_y_start = max(0, -dy)
        src_y_end = src_y_start + (y_end - y_start)
        src_x_start = max(0, -dx)
        src_x_end = src_x_start + (x_end - x_start)
        shifted[z_start:z_end, y_start:y_end, x_start:x_end] = mask[src_z_start:src_z_end, src_y_start:src_y_end, src_x_start:src_x_end]
        result |= shifted
    return result


def _spiral_sample_sphere(n_samps: int) -> np.ndarray:
    n_samps = max(4, int(n_samps))
    indices = np.arange(0, n_samps, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_samps)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)

def _spiral_sample_sphere_with_tris(n_samps: int) -> tuple[np.ndarray, np.ndarray]:
    V = _spiral_sample_sphere(n_samps)
    Tri = np.zeros((0, 3), dtype=np.int32)
    try:
        from scipy.spatial import ConvexHull  # type: ignore

        hull = ConvexHull(V)
        Tri = hull.simplices.astype(np.int32)
    except Exception:
        # If SciPy is unavailable fall back to empty triangulation rather than silently zeroing.
        Tri = np.zeros((0, 3), dtype=np.int32)
    return V, Tri

def _teardrop_poj(points: np.ndarray, mode: int) -> np.ndarray:
    out = points.copy()
    if mode == 1:
        out[:, 2] = np.sign(out[:, 2]) * (np.abs(out[:, 2]) ** 1.2)
    elif mode == 2:
        out[:, 0] = np.sign(out[:, 0]) * (np.abs(out[:, 0]) ** 1.1)
        out[:, 1] = np.sign(out[:, 1]) * (np.abs(out[:, 1]) ** 1.1)
    return out


def _rotation_matrix(deg: float, axis: str) -> np.ndarray:
    rad = np.deg2rad(deg)
    c = np.cos(rad)
    s = np.sin(rad)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    raise ValueError("Unknown axis")


def _pos_to_dists(positions: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances for capillary positions."""

    diffs = positions[:, None, :] - positions[None, :, :]
    return np.sqrt((diffs**2).sum(axis=2))


def _lin_to_subs(lin: np.ndarray, shape: tuple[int, ...]) -> tuple[np.ndarray, ...]:
    lin = np.asarray(lin, dtype=np.int64).ravel()
    return np.unravel_index(lin - 1, shape, order="F")


def _get_lin(arr: np.ndarray, lin: np.ndarray) -> np.ndarray:
    subs = _lin_to_subs(lin, arr.shape)
    return arr[subs]


def _set_lin(arr: np.ndarray, lin: np.ndarray, values: Any) -> None:
    subs = _lin_to_subs(lin, arr.shape)
    arr[subs] = values


def _intriangulation(vertices: np.ndarray, faces: np.ndarray, testp: np.ndarray) -> np.ndarray:
    """Ray casting test for points inside a closed triangulation."""

    if faces.size == 0 or testp.size == 0:
        return np.zeros((testp.shape[0],), dtype=bool)

    verts = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=int)
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0

    dir_vec = np.array([1.0, 0.0, 0.0], dtype=float)
    h = np.cross(dir_vec, e2)
    a = np.einsum("ij,ij->i", e1, h)
    eps = 1e-10

    inside = np.zeros(testp.shape[0], dtype=bool)
    for idx, orig in enumerate(np.asarray(testp, dtype=float)):
        valid = np.abs(a) > eps
        if not np.any(valid):
            continue
        f = np.zeros_like(a)
        f[valid] = 1.0 / a[valid]
        s = orig - v0
        u = f * np.einsum("ij,ij->i", s, h)
        valid = valid & (u >= 0.0) & (u <= 1.0)
        q = np.cross(s, e1)
        v = f * q[:, 0]
        valid = valid & (v >= 0.0) & (u + v <= 1.0)
        t = f * np.einsum("ij,ij->i", e2, q)
        valid = valid & (t > eps)
        inside[idx] = (np.count_nonzero(valid) % 2) == 1
    return inside


@dataclass
class VolumeOutputs:
    neur_vol: np.ndarray
    gp_nuc: np.ndarray
    gp_soma: np.ndarray
    gp_vals: np.ndarray
    neur_ves: np.ndarray
    bg_proc: np.ndarray
    neur_ves_all: np.ndarray
    locs: np.ndarray
    neur_num_AD: np.ndarray | None = None
    neur_soma: np.ndarray | None = None
    neur_num: np.ndarray | None = None
    gp_bgvals: np.ndarray | None = None
    Vcell: np.ndarray | None = None
    Vnuc: np.ndarray | None = None
    Tri: np.ndarray | None = None


def check_vol_params(params: Dict[str, Any] | None) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "vol_sz": [100, 100, 50],
        "min_dist": 16,
        "vres": 2,
        "N_bg": 1e6,
        "vol_depth": 200,
        "dendrite_tau": 5,
        "verbose": 1,
    }
    params = params or {}
    out = defaults | params
    out["vol_sz"] = list(out["vol_sz"])
    out["vol_sz"][2] = int(np.ceil(out["vol_sz"][2] / 10.0) * 10)
    if "N_neur" not in out or out["N_neur"] is None:
        density = out.get("neur_density", 1e5)
        out["neur_density"] = density
        out["N_neur"] = int(np.ceil(density * np.prod(out["vol_sz"]) / 1e9))
    elif "neur_density" not in out or out["neur_density"] is None:
        out["neur_density"] = 1e9 * out["N_neur"] / np.prod(out["vol_sz"])

    if "N_den" not in out or out["N_den"] is None:
        if "AD_density" in out and out["AD_density"] is not None:
            out["N_den"] = out["AD_density"] * np.prod(out["vol_sz"][:2]) / 1e6
        else:
            out["AD_density"] = 2e3
            out["N_den"] = out["AD_density"] * np.prod(out["vol_sz"][:2]) / 1e6
    return out


def check_neur_params(params: Dict[str, Any] | None) -> Dict[str, Any]:
    defaults = {
        "n_samps": 200,
        "l_scale": 90,
        "p_scale": 1000,
        "avg_rad": 5.9,
        "nuc_rad": [5.65, 2.5],
        "max_ang": 20,
        "plot_opt": False,
        "dendrite_tau": 50,
        "nuc_fluorsc": 0,
        "min_thic": [0.4, 0.4],
        "eccen": [0.35, 0.35, 0.5],
        "exts": [0.75, 1.7],
        "nexts": [0.5, 1.0],
        "neur_type": "pyr",
        "fluor_dist": [1, 0.2],
    }
    params = params or {}
    return {**defaults, **params}


def check_vasc_params(params: Dict[str, Any] | None) -> Dict[str, Any]:
    defaults = {
        "flag": 1,
        "ves_shift": [5, 15, 5],
        "depth_vasc": 200,
        "depth_surf": 15,
        "distWeightScale": 2,
        "randWeightScale": 0.1,
        "cappAmpScale": 0.5,
        "cappAmpZscale": 0.5,
        "vesSize": [15, 9, 2],
        "vesFreq": [125, 200, 50],
        "sourceFreq": 1000,
        "vesNumScale": 0.2,
        "sepweight": 0.75,
        "distsc": 4,
        "node_params": {
            "maxit": 25,
            "lensc": 50,
            "varsc": 15,
            "mindist": 10,
            "varpos": 5,
            "dirvar": np.pi / 8,
            "branchp": 0.02,
            "vesrad": 25,
        },
    }
    params = params or {}
    merged = defaults | params
    merged["node_params"] = {**defaults["node_params"], **params.get("node_params", {})}
    return merged


def check_dend_params(params: Dict[str, Any] | None) -> Dict[str, Any]:
    defaults = {
        "dtParams": [40, 150, 50, 1, 10],
        "atParams": [6, 5, 5, 5, 1],
        "atParams2": [1, 5, 5, 5, 4],
        "dweight": 10,
        "bweight": 5,
        "thicknessScale": 0.5,
        "weightScale": [150, 1, 0.8],
        "dims": [60, 60, 60],
        "dimsSS": [5, 5, 5],
        "rallexp": 1.5,
    }
    params = params or {}
    return {**defaults, **params}


def check_bg_params(params: Dict[str, Any] | None) -> Dict[str, Any]:
    defaults = {
        "flag": 1,
        "distsc": 0.5,
        "fillweight": 100,
        "maxlength": 200,
        "minlength": 10,
        "maxdist": 100,
        "maxel": 1,
    }
    params = params or {}
    return {**defaults, **params}


def check_axon_params(params: Dict[str, Any] | None) -> Dict[str, Any]:
    defaults = {
        "flag": 1,
        "distsc": 0.5,
        "fillweight": 100,
        "maxlength": 200,
        "minlength": 10,
        "maxdist": 100,
        "maxel": 8,
        "varfill": 0.3,
        "maxvoxel": 6,
        "padsize": 20,
        "numbranches": 20,
        "varbranches": 5,
        "maxfill": 0.5,
        "N_proc": 10,
        "l": 25,
        "rho": 0.1,
    }
    params = params or {}
    return {**defaults, **params}


def generate_connection(
    start: Any = None,
    ends: Any = None,
    weight: Any = None,
    locs: Any = None,
    misc: Any = None,
) -> Dict[str, Any]:
    return {
        "start": start if start is not None else [],
        "ends": ends if ends is not None else [],
        "weight": weight if weight is not None else [],
        "locs": locs if locs is not None else [],
        "misc": misc if misc is not None else [],
    }


def generate_node(
    num: Any = None,
    root: Any = None,
    conn: Any = None,
    pos: Any = None,
    node_type: Any = None,
    misc: Any = None,
) -> Dict[str, Any]:
    return {
        "num": num if num is not None else [],
        "root": root if root is not None else [],
        "conn": conn if conn is not None else [],
        "pos": np.asarray(pos, dtype=float) if pos is not None else [],
        "type": node_type if node_type is not None else [],
        "misc": misc if misc is not None else [],
    }


def simulate_blood_vessels(vol_params: Dict[str, Any], vasc_params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
    """Translate simulatebloodvessels.m to Python."""

    vol_params = check_vol_params(vol_params)
    vasc_params = check_vasc_params(vasc_params)

    if vol_params.get("verbose", 1) == 1:
        print("Generating in-volume blood vessels...", end="")
    elif vol_params.get("verbose", 1) > 1:
        print("Generating in-volume blood vessels...")
        start_time = time.time()

    vres = float(vol_params["vres"])
    vp = dict(vasc_params)
    vp["depth_surf"] = float(vasc_params["depth_surf"]) * vres
    vp["mindists"] = np.array(vasc_params["vesFreq"], dtype=float) * vres / 2.0
    vp["maxcappdist"] = 2 * float(vasc_params["vesFreq"][2]) * vres
    vp["vesSize"] = np.array(vasc_params["vesSize"], dtype=float) * vres

    node_params = vasc_params["node_params"]
    np_nodes = dict(node_params)
    np_nodes["lensc"] = float(node_params["lensc"]) * vres
    np_nodes["varsc"] = float(node_params["varsc"]) * vres
    np_nodes["mindist"] = float(node_params["mindist"]) * vres
    np_nodes["varpos"] = float(node_params["varpos"]) * vres
    np_nodes["vesrad"] = float(node_params["vesrad"]) * vres

    if not vol_params.get("vasc_sz"):
        nv_vol_sz = np.array(vol_params["vol_sz"], dtype=float) + np.array([0.0, 0.0, float(vol_params["vol_depth"])])
    else:
        nv_vol_sz = np.array(vol_params["vasc_sz"], dtype=float)

    nv = {
        "vol_sz": nv_vol_sz,
        "size": (nv_vol_sz * vres).astype(int),
        "szum": nv_vol_sz.astype(int),
    }

    rng = np.random.default_rng(int(vol_params.get("seed", 0)) + 7)
    nv["nsource"] = max(
        int(round((2 * (nv_vol_sz[0] + nv_vol_sz[1]) / vp["sourceFreq"]) * abs(1 + vp["vesNumScale"] * rng.standard_normal()))),
        0,
    )
    nv["nvert"] = max(
        int(round((nv_vol_sz[0] * nv_vol_sz[1] / (vp["vesFreq"][1] ** 2)) * abs(1 + vp["vesNumScale"] * rng.standard_normal()))),
        0,
    )
    nv["nsurf"] = max(
        int(round((nv_vol_sz[0] * nv_vol_sz[1] / (vp["vesFreq"][0] ** 2)) * abs(1 + vp["vesNumScale"] * rng.standard_normal()))),
        0,
    )
    nv["ncapp"] = max(
        int(round((np.prod(nv_vol_sz) / (vp["vesFreq"][2] ** 3)) * abs(1 + vp["vesNumScale"] * rng.standard_normal()))),
        0,
    )

    nodes, nv = grow_major_vessels(nv, np_nodes, vp)
    conn = nodes_to_conn(nodes)
    nv["nconn"] = len(conn)

    for edge in conn:
        start = int(edge.get("start", 0))
        ends = int(edge.get("ends", 0))
        for node_idx in (start, ends):
            if not (0 <= node_idx < len(nodes)):
                continue
            if nodes[node_idx].get("type") in {"edge", "surf", "sfvt"}:
                node_pos = np.asarray(nodes[node_idx]["pos"], dtype=float)
                denom = max(1, len(nodes[node_idx].get("conn", [])))
                weight = float(edge.get("weight") or 0.0)
                node_pos[2] = min(node_pos[2] + np.ceil(weight / denom), nv["size"][2])
                nodes[node_idx]["pos"] = node_pos

    neur_ves, conn = conn_to_vol(nodes, conn, nv)
    nodes, conn, nv = grow_capillaries(nodes, conn, neur_ves, nv, vp, vres)

    cappidxs = [i for i, c in enumerate(conn) if not c.get("locs")]
    neur_ves, _ = conn_to_vol(nodes, conn, nv, cappidxs, neur_ves)

    if vol_params.get("vasc_sz"):
        neur_ves_all = neur_ves
        vol_sz = np.array(vol_params["vol_sz"], dtype=int)
        vol_depth = int(vol_params["vol_depth"])
        sz = np.array([vol_sz[0], vol_sz[1], vol_depth + vol_sz[2]], dtype=int) * int(vres)
        sz_diff = np.ceil((np.array(vol_params["vasc_sz"], dtype=float) * vres - sz) / 2.0).astype(int)
        neur_ves = neur_ves[
            sz_diff[0] : sz_diff[0] + sz[0],
            sz_diff[1] : sz_diff[1] + sz[1],
            sz_diff[2] : sz_diff[2] + sz[2],
        ]
    else:
        neur_ves_all = neur_ves

    if vol_params.get("verbose", 1) == 1:
        print("done.")
    elif vol_params.get("verbose", 1) > 1:
        elapsed = time.time() - start_time
        print(f"done ({elapsed:.3f} seconds).")

    return neur_ves.astype(bool), vasc_params, neur_ves_all.astype(bool)


def sample_dense_neurons(
    neur_params: Dict[str, Any], vol_params: Dict[str, Any], neur_ves: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Translate sampleDenseNeurons.m with simplified placement logic."""

    vol_params = check_vol_params(vol_params)
    neur_params = check_neur_params(neur_params)
    eta = 1.1

    min_dist = float(vol_params["min_dist"])
    radius = int(np.ceil(min_dist / 2))
    grid = np.arange(-radius, radius + 1)
    x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")
    se = (x**2 + y**2 + z**2) <= (min_dist / 2) ** 2
    neur_ves_trunc = _binary_dilate_3d(neur_ves.astype(bool), se)

    V, Tri = _spiral_sample_sphere_with_tris(int(neur_params["n_samps"]))
    neur_params["S_samp"] = V
    neur_params["Tri"] = Tri

    Vcell = np.zeros((V.shape[0], 3, 0), dtype=np.float32)
    Vnuc = np.zeros((V.shape[0], 3, 0), dtype=np.float32)
    rot_ang = np.zeros((0, 3), dtype=np.float32)

    vol_sz = np.array(vol_params["vol_sz"], dtype=int)
    vres = int(vol_params["vres"])
    mesh_x, mesh_y, mesh_z = np.meshgrid(
        np.linspace(0, vol_sz[0], vol_sz[0] * vres),
        np.linspace(0, vol_sz[1], vol_sz[1] * vres),
        np.linspace(0, vol_sz[2], vol_sz[2] * vres),
        indexing="ij",
    )
    mesh_x = np.swapaxes(mesh_x, 0, 1)
    mesh_y = np.swapaxes(mesh_y, 0, 1)
    mesh_z = np.swapaxes(mesh_z, 0, 1)

    vol_depth = int(vol_params["vol_depth"] * vres)
    idx_good = neur_ves_trunc[:, :, vol_depth : vol_depth + vol_sz[2] * vres]
    idx_good = ~idx_good
    idx_bad = idx_good.copy()

    if vol_params.get("verbose", 1) == 1:
        print("Sampling random locations for the neurons...", end="")
    elif vol_params.get("verbose", 1) > 1:
        print("Sampling random locations for the neurons...")

    neur_locs = np.array([[np.inf, np.inf, np.inf]], dtype=float)
    while np.sum(idx_good) > 1 and Vcell.shape[2] < int(vol_params["N_neur"]):
        V_tmp, Vnuc_tmp, _, rot_tmp = generate_neural_body(neur_params)
        Vcell = np.concatenate([Vcell, V_tmp[:, :, None]], axis=2)
        Vnuc = np.concatenate([Vnuc, Vnuc_tmp[:, :, None]], axis=2)
        rot_ang = np.vstack([rot_ang, rot_tmp])

        flat_idx = np.flatnonzero(idx_good.ravel(order="F"))
        if flat_idx.size == 0:
            flat_idx = np.flatnonzero(idx_bad.ravel(order="F"))
        if flat_idx.size == 0:
            break
        idx_now = np.random.choice(flat_idx)
        new_pt = np.array(
            [
                mesh_x.ravel(order="F")[idx_now],
                mesh_y.ravel(order="F")[idx_now],
                mesh_z.ravel(order="F")[idx_now],
            ]
        )
        if int(vol_params["N_neur"]) == 1:
            new_pt = np.round(vol_sz / 2)

        neur_locs = np.vstack([neur_locs, new_pt])
        dist = np.sqrt((mesh_x - new_pt[0]) ** 2 + (mesh_y - new_pt[1]) ** 2 + (mesh_z - new_pt[2]) ** 2)
        idx_good[dist <= eta * min_dist] = False
        idx_bad[dist <= min_dist] = False
        idx_good = ~(idx_good | ~idx_bad)

        if vol_params.get("verbose", 1) == 1:
            print(".", end="")

    neur_locs = neur_locs[1:].astype(np.float32)
    n_neur = Vcell.shape[2]
    Vcell = (Vcell + neur_locs.T.reshape(1, 3, n_neur)).astype(np.float32)
    Vnuc = (Vnuc + neur_locs.T.reshape(1, 3, n_neur)).astype(np.float32)

    if vol_params.get("verbose", 1) >= 1:
        print("done.")

    return neur_locs, Vcell, Vnuc, Tri, rot_ang.astype(np.float32)


def generate_neural_body(neur_params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    neur_params = check_neur_params(neur_params)
    rng = np.random.default_rng(int(neur_params.get("seed", 0)))
    pwr = 1.0
    nuc_offset = 3.0

    if neur_params.get("S_samp") is None and neur_params.get("Tri") is None:
        V = _spiral_sample_sphere(int(neur_params["n_samps"]))
        Tri = np.zeros((0, 3), dtype=np.int32)
    else:
        V = np.asarray(neur_params.get("S_samp"), dtype=float)
        Tri = np.asarray(neur_params.get("Tri"), dtype=np.int32)

    if neur_params.get("neur_type") == "pyr":
        Vtear = _teardrop_poj(V, 1)
    elif neur_params.get("neur_type") == "peanut":
        Vtear = _teardrop_poj(V, 2)
    else:
        Vtear = V

    if neur_params.get("dists") is None:
        diffs = V[:, None, :] - V[None, :, :]
        dists = np.sqrt((diffs**2).sum(axis=2))
        dists = 2 * np.arcsin(dists / 2)
        dists = neur_params["p_scale"] * np.exp(-(dists / neur_params["l_scale"]) ** pwr)
    else:
        dists = np.asarray(neur_params.get("dists"), dtype=float)
        dists = neur_params["p_scale"] * np.exp(-(dists / neur_params["l_scale"]) ** pwr)

    if neur_params.get("Rtear") is None:
        if neur_params.get("neur_type") == "pyr":
            Rtear = np.sqrt((Vtear**2).sum(axis=1))
        elif neur_params.get("neur_type") == "peanut":
            Rtear = np.sqrt((V**2).sum(axis=1))
        else:
            Rtear = 1.0
    else:
        Rtear = np.asarray(neur_params.get("Rtear"), dtype=float)

    try:
        min_eig = 1.03 * np.linalg.eigvalsh(dists).min()
    except np.linalg.LinAlgError:
        min_eig = 0.0
    if min_eig < 0:
        dists = dists + abs(min_eig) * np.eye(dists.shape[0])

    x_bnds = np.array(neur_params["exts"], dtype=float) * neur_params["avg_rad"]
    x_base = np.abs(rng.multivariate_normal(np.zeros(len(Rtear)), dists))
    x = x_base - x_base.mean() + neur_params["avg_rad"] * np.asarray(Rtear)
    xmin = min(x.min(), x_bnds[0])
    x = (x_bnds[1] - x_bnds[0]) * (x - xmin) / (max(x.max(), x_bnds[1]) - xmin) + x_bnds[0]

    if neur_params.get("neur_type") == "pyr":
        x2 = x_base - x_base.mean() + neur_params["avg_rad"]
        xmin2 = min(x2.min(), x_bnds[0])
        x2 = (x_bnds[1] - x_bnds[0]) * (x2 - xmin2) / (max(x2.max(), x_bnds[1]) - xmin2) + x_bnds[0]
    else:
        x2 = x.copy()

    eccens = np.ones(3) + np.asarray(neur_params["eccen"]) * (rng.random(3) - np.array([0.5, 0.5, 0.0]))
    eccens = eccens / (np.prod(eccens) ** (1 / 3))

    if neur_params.get("neur_type") == "pyr":
        Vetear = Vtear * eccens
        Vetear = Vetear / np.sqrt(np.mean((Vetear**2).sum(axis=1)))
    else:
        Vetear = V * eccens
        Vetear = Vetear / np.sqrt(np.mean((V**2).sum(axis=1)))

    Vcell = Vetear * x[:, None]
    Vcell = Vcell + np.array([0.0, 0.0, -nuc_offset])
    vnorms = np.sqrt((Vcell**2).sum(axis=1))

    Vnuc = (V * np.array([1.0, 1.0, -1.0])) * x2[:, None]
    vnorms2 = np.sqrt((Vnuc**2).sum(axis=1))
    vnorms2 = neur_params["nexts"][1] * (
        neur_params["nexts"][0] * (vnorms2 - vnorms2.min()) + (1 - neur_params["nexts"][0]) * vnorms2.max()
    )
    vnorms2 = vnorms2 + min(vnorms - vnorms2) - neur_params["min_thic"][0]
    Vnuc = (Vnuc * eccens) * (vnorms2 / np.sqrt((Vnuc**2).sum(axis=1)))[:, None]

    lat_ang = rng.random() * 2 * np.pi
    lat_shift = (1 - abs(rng.random() - rng.random())) * neur_params["min_thic"][1] * np.array(
        [np.sin(lat_ang), np.cos(lat_ang)]
    )
    Vcell = Vcell + np.array([0.0, 0.0, nuc_offset])
    Vnuc = Vnuc + np.array([lat_shift[0], lat_shift[1], nuc_offset])

    if neur_params.get("nuc_rad") is not None:
        nuc_rad = np.asarray(neur_params["nuc_rad"], dtype=float)
        nuc_sz = (4 / 3) * np.pi * (nuc_rad[0] ** 3)
        current_sz = 0.0
        try:
            from scipy.spatial import ConvexHull  # type: ignore

            hull = ConvexHull(Vnuc)
            current_sz = float(hull.volume)
        except Exception:
            vnorms2 = np.sqrt((Vnuc**2).sum(axis=1))
            current_sz = (4 / 3) * np.pi * (vnorms2.mean() ** 3)
        if current_sz > 0:
            scale = (nuc_sz / current_sz) ** (1 / 3)
            if len(nuc_rad) > 1:
                scale = scale ** (1 / nuc_rad[1])
            Vnuc = Vnuc * scale

    max_ang = neur_params.get("max_ang", 20)
    angles = -abs(max_ang) + 2 * abs(max_ang) * rng.random(3)
    Rx = _rotation_matrix(angles[0], "x")
    Ry = _rotation_matrix(angles[1], "y")
    Rz = _rotation_matrix(angles[2], "z")
    Vnuc = Vnuc @ Rx @ Ry @ Rz
    Vcell = Vcell @ Rx @ Ry @ Rz

    return Vcell.astype(np.float32), Vnuc.astype(np.float32), Tri.astype(np.int32), angles.astype(np.float32)


def generate_neural_volume(
    neur_params: Dict[str, Any],
    vol_params: Dict[str, Any],
    neur_locs: np.ndarray,
    Vcell: np.ndarray,
    Vnuc: np.ndarray,
    neur_ves: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, list, list]:
    vol_params = check_vol_params(vol_params)
    neur_params = check_neur_params(neur_params)

    if vol_params.get("verbose", 1) >= 1:
        print("Setting up volume...", end="")

    vol_sz = np.array(vol_params["vol_sz"], dtype=int)
    vres = int(vol_params["vres"])
    vol_shape = tuple(vol_sz * vres)
    neur_soma = np.zeros(vol_shape, dtype=np.uint16)
    neur_vol = np.zeros(vol_shape, dtype=np.float32)
    gp_nuc = [[None, None] for _ in range(int(vol_params["N_neur"]))]
    gp_soma = [[None] for _ in range(int(vol_params["N_neur"]))]

    taken_pts = neur_ves.copy()
    vol_depth = int(vol_params["vol_depth"] * vres)
    if taken_pts.shape != vol_shape and taken_pts.shape[2] >= vol_depth + vol_shape[2]:
        taken_pts = taken_pts[:, :, vol_depth : vol_depth + vol_shape[2]]

    if neur_params.get("Tri") is None or np.asarray(neur_params.get("Tri")).size == 0:
        _, Tri = _spiral_sample_sphere_with_tris(int(neur_params["n_samps"]))
    else:
        Tri = np.asarray(neur_params.get("Tri"), dtype=int)
    if Tri.size and Tri.min() == 1:
        Tri = Tri - 1

    if vol_params.get("verbose", 1) >= 1:
        print("done.")
        print("Finding interior points...")

    for kk in range(int(vol_params["N_neur"])):
        max_ext = np.ceil(np.max(np.linalg.norm(Vcell[:, :, kk] - neur_locs[kk], axis=1)))
        idx_pos = np.round(vres * neur_locs[kk]).astype(int)
        idxX = np.arange(
            max(1, int(idx_pos[0] - vres * max_ext)), min(int(idx_pos[0] + vres * max_ext), int(vol_sz[0] * vres)) + 1, dtype=int
        )
        idxY = np.arange(
            max(1, int(idx_pos[1] - vres * max_ext)), min(int(idx_pos[1] + vres * max_ext), int(vol_sz[1] * vres)) + 1, dtype=int
        )
        idxZ = np.arange(
            max(1, int(idx_pos[2] - vres * max_ext)), min(int(idx_pos[2] + vres * max_ext), int(vol_sz[2] * vres)) + 1, dtype=int
        )

        mesh_x, mesh_y, mesh_z = np.meshgrid(
            (idxX - idx_pos[0]) / vres,
            (idxY - idx_pos[1]) / vres,
            (idxZ - idx_pos[2]) / vres,
            indexing="ij",
        )
        dist = np.sqrt(mesh_x**2 + mesh_y**2 + mesh_z**2)
        idx_to_test = dist <= max_ext

        idx_tri = np.stack(
            [
                mesh_x[idx_to_test] + idx_pos[0] / vres + 1 / vres / 2,
                mesh_y[idx_to_test] + idx_pos[1] / vres + 1 / vres / 2,
                mesh_z[idx_to_test] + idx_pos[2] / vres + 1 / vres / 2,
            ],
            axis=1,
        )

        tmp1 = _intriangulation(Vcell[:, :, kk], Tri, idx_tri)
        tmp = _intriangulation(Vnuc[:, :, kk], Tri, idx_tri)

        tmp1a = np.zeros_like(idx_to_test, dtype=bool)
        tmp1a[idx_to_test] = tmp1
        tmpa = np.zeros_like(idx_to_test, dtype=bool)
        tmpa[idx_to_test] = tmp

        neur_idx = tmp1a & (~tmpa)
        neur_idx = neur_idx & (~taken_pts[np.ix_(idxX - 1, idxY - 1, idxZ - 1)])
        taken_pts[np.ix_(idxX - 1, idxY - 1, idxZ - 1)] |= neur_idx

        iX, iY, iZ = np.nonzero(neur_idx)
        iX = iX + idxX[0] - 1
        iY = iY + idxY[0] - 1
        iZ = iZ + idxZ[0] - 1
        neur_idx2 = np.ravel_multi_index((iX, iY, iZ), vol_shape, order="F") + 1
        _set_lin(neur_soma, neur_idx2, kk + 1)
        gp_soma[kk][0] = neur_idx2.astype(np.int32)

        nuc_iX, nuc_iY, nuc_iZ = np.nonzero(tmpa)
        nuc_iX = nuc_iX + idxX[0] - 1
        nuc_iY = nuc_iY + idxY[0] - 1
        nuc_iZ = nuc_iZ + idxZ[0] - 1
        nuc_idx = np.ravel_multi_index((nuc_iX, nuc_iY, nuc_iZ), vol_shape, order="F") + 1
        gp_nuc[kk][0] = nuc_idx.astype(np.int32)
        gp_nuc[kk][1] = neur_params["nuc_fluorsc"]
        _set_lin(neur_vol, nuc_idx, neur_params["nuc_fluorsc"])

        if vol_params.get("verbose", 1) == 1:
            print(".", end="")
        elif vol_params.get("verbose", 1) > 1:
            print(f"{kk + 1} done.")

    if vol_params.get("verbose", 1) >= 1:
        print("done.")

    return neur_soma, neur_vol, gp_nuc, gp_soma


def grow_neuron_dendrites(
    vol_params: Dict[str, Any],
    dend_params: Dict[str, Any],
    neur_soma: np.ndarray,
    neur_ves: np.ndarray,
    neur_locs: np.ndarray,
    gp_nuc: np.ndarray,
    gp_soma: np.ndarray,
    rotAng: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], np.ndarray]:
    if rotAng is None:
        rotAng = np.zeros((neur_locs.shape[0], 3), dtype=np.float32)

    vol_params = check_vol_params(vol_params)
    dend_params = check_dend_params(dend_params)
    rng = np.random.default_rng(int(dend_params.get("seed", 0)) + 2)

    if vol_params.get("verbose", 1) == 1:
        print("Growing out dendrites", end="")
    elif vol_params.get("verbose", 1) > 1:
        print("Growing out dendrites...")

    dt_params = list(dend_params["dtParams"])
    at_params = list(dend_params["atParams"])
    dweight = float(dend_params["dweight"])
    bweight = float(dend_params["bweight"])
    thickness_scale = float(dend_params["thicknessScale"])
    dims = np.array(dend_params["dims"], dtype=int)
    dims_ss = np.array(dend_params["dimsSS"], dtype=int)
    rallexp = float(dend_params["rallexp"])
    vres = int(vol_params["vres"])
    n_neur = int(vol_params["N_neur"])
    vol_sz = np.array(vol_params["vol_sz"], dtype=int)
    dims = np.minimum(dims, (vol_sz / dims_ss).astype(int))
    fulldims = vol_sz * vres
    dims = dims * vres
    dt_params[1] *= vres
    dt_params[2] *= vres
    at_params[1] *= vres
    at_params[2] *= vres
    at_params[3] *= vres
    thickness_scale = thickness_scale * vres * vres

    vol_depth = int(vol_params["vol_depth"] * vres)
    cell_volume = neur_soma.astype(np.float32) + (
        vol_params["N_den"] + vol_params["N_neur"] + vol_params["N_bg"] + 1
    ) * neur_ves[:, :, vol_depth : vol_depth + vol_sz[2] * vres].astype(np.float32)
    for kk in range(n_neur):
        if kk < len(gp_nuc) and gp_nuc[kk][0] is not None:
            _set_lin(cell_volume, np.array(gp_nuc[kk][0], dtype=int), kk + 1)

    if gp_soma is None or not gp_soma:
        gp_soma = [[np.flatnonzero(neur_soma == (kk + 1)) + 1] for kk in range(n_neur)]

    neur_num = cell_volume.astype(np.uint16)
    cell_volume_idx = np.zeros_like(neur_soma, dtype=np.float32)
    cell_volume_val = np.zeros_like(neur_soma, dtype=np.float32)
    cell_volume_ad = np.zeros_like(neur_soma, dtype=bool)
    allroots = np.ceil(np.maximum(vres * neur_locs, 1e-4)).astype(int)
    fdims = np.minimum(dims * dims_ss, fulldims)
    ML = np.full((*fdims, 6), np.inf, dtype=np.float32)
    dend_var = float(dend_params.get("dendVar", 0.25))

    small_z = fulldims[2] <= fdims[2]
    for j in range(n_neur):
        if vol_params.get("verbose", 1) > 1:
            start_t = time.time()

        aproot = np.unravel_index(int(np.min(gp_soma[j][0])) - 1, tuple(fulldims), order="F")
        numdt = max(1, int(dt_params[0] + round(dt_params[4] * rng.standard_normal())))

        borderflag = False
        try:
            if small_z:
                rootL = np.array([fdims[0] / 2 + 1, fdims[1] / 2 + 1, allroots[j, 2]])
                obstruction = cell_volume[
                    allroots[j, 0] - fdims[0] // 2 : allroots[j, 0] + fdims[0] // 2,
                    allroots[j, 1] - fdims[1] // 2 : allroots[j, 1] + fdims[1] // 2,
                    :,
                ]
            else:
                rootL = np.array([fdims[0] / 2 + 1, fdims[1] / 2 + 1, fdims[2] / 2 + 1])
                obstruction = cell_volume[
                    allroots[j, 0] - fdims[0] // 2 : allroots[j, 0] + fdims[0] // 2,
                    allroots[j, 1] - fdims[1] // 2 : allroots[j, 1] + fdims[1] // 2,
                    allroots[j, 2] - fdims[2] // 2 : allroots[j, 2] + fdims[2] // 2,
                ]
            if obstruction.shape != tuple(fdims):
                raise ValueError("obstruction shape mismatch")
        except Exception:
            obstruction = np.zeros(fdims, dtype=np.float32)
            borderflag = True
            if small_z:
                rootL = np.array([fdims[0] / 2 + 1, fdims[1] / 2 + 1, allroots[j, 2]])
                Xlims = np.array(
                    [allroots[j, 0] - fdims[0] / 2, allroots[j, 0] + fdims[0] / 2 - 1], dtype=int
                )
                Ylims = np.array(
                    [allroots[j, 1] - fdims[1] / 2, allroots[j, 1] + fdims[1] / 2 - 1], dtype=int
                )
                xs = slice(max(1, Xlims[0]) - 1, min(Xlims[1], fulldims[0]))
                ys = slice(max(1, Ylims[0]) - 1, min(Ylims[1], fulldims[1]))
                xdst = slice(max(1, -Xlims[0] + 2) - 1, fdims[0] + min(0, -Xlims[1] + fulldims[0]))
                ydst = slice(max(1, -Ylims[0] + 2) - 1, fdims[1] + min(0, -Ylims[1] + fulldims[1]))
                obstruction[xdst, ydst, :] = cell_volume[xs, ys, :]
            else:
                rootL = np.array([fdims[0] / 2 + 1, fdims[1] / 2 + 1, fdims[2] / 2 + 1])
                Xlims = np.array(
                    [allroots[j, 0] - fdims[0] / 2, allroots[j, 0] + fdims[0] / 2 - 1], dtype=int
                )
                Ylims = np.array(
                    [allroots[j, 1] - fdims[1] / 2, allroots[j, 1] + fdims[1] / 2 - 1], dtype=int
                )
                Zlims = np.array(
                    [allroots[j, 2] - fdims[2] / 2, allroots[j, 2] + fdims[2] / 2 - 1], dtype=int
                )
                xs = slice(max(1, Xlims[0]) - 1, min(Xlims[1], fulldims[0]))
                ys = slice(max(1, Ylims[0]) - 1, min(Ylims[1], fulldims[1]))
                zs = slice(max(1, Zlims[0]) - 1, min(Zlims[1], fulldims[2]))
                xdst = slice(max(1, -Xlims[0] + 2) - 1, fdims[0] + min(0, -Xlims[1] + fulldims[0]))
                ydst = slice(max(1, -Ylims[0] + 2) - 1, fdims[1] + min(0, -Ylims[1] + fulldims[1]))
                zdst = slice(max(1, -Zlims[0] + 2) - 1, fdims[2] + min(0, -Zlims[1] + fulldims[2]))
                obstruction[xdst, ydst, zdst] = cell_volume[xs, ys, zs]

        cell_body = np.flatnonzero(obstruction == (j + 1))
        if cell_body.size:
            obstruction.flat[cell_body] = 0

        root = np.ceil(rootL / dims_ss).astype(int)
        root = np.minimum(root, dims)
        M = 1 + dweight * rng.random((*dims, 6), dtype=np.float32)

        aprootS = root + np.round((np.array(aproot) - allroots[j]) / dims_ss).astype(int)
        aprootS = np.clip(aprootS, 1, dims)
        if aprootS[0] > root[0]:
            M[root[0] - 1 : aprootS[0], root[1] - 1, root[2] - 1, 0] = 0
        elif aprootS[0] < root[0]:
            M[aprootS[0] - 1 : root[0], root[1] - 1, root[2] - 1, 1] = 0
        if aprootS[1] > root[1]:
            M[aprootS[0] - 1, root[1] - 1 : aprootS[1], root[2] - 1, 2] = 0
        elif aprootS[1] < root[1]:
            M[aprootS[0] - 1, aprootS[1] - 1 : root[1], root[2] - 1, 3] = 0
        if aprootS[2] > root[2]:
            M[aprootS[0] - 1, aprootS[1] - 1, root[2] - 1 : aprootS[2], 4] = 0
        elif aprootS[2] < root[2]:
            M[aprootS[0] - 1, aprootS[1] - 1, aprootS[2] - 1 : root[2], 5] = 0

        M[0, :, :, 0] = np.inf
        M[-1, :, :, 1] = np.inf
        M[:, 0, :, 2] = np.inf
        M[:, -1, :, 3] = np.inf
        M[:, :, 0, 4] = np.inf
        M[:, :, -1, 5] = np.inf

        fillfrac = obstruction > 0
        fillfrac = fillfrac.reshape(dims_ss[0], dims[0], dims_ss[1], dims[1], dims_ss[2], dims[2])
        fillfrac = np.sum(fillfrac, axis=(0, 2, 4)) / np.prod(dims_ss)
        M = M + (-bweight * np.log(1 - (2 * np.maximum(0, fillfrac - 0.5))))[:, :, :, None]

        pathfrom = dendrite_dijkstra2(M.reshape(-1, 6, order="F"), tuple(dims), tuple(root))[1]

        ends_t = np.zeros((numdt, 3), dtype=int)
        for i in range(numdt):
            for _ in range(100):
                theta = rng.random() * 2 * np.pi
                r = np.sqrt(rng.random()) * dt_params[1]
                end_t = np.floor(
                    np.array([r * np.cos(theta) + rootL[0], r * np.sin(theta) + rootL[1], 2 * dt_params[2] * (rng.random() - 0.5) + rootL[2]])
                ).astype(int)
                end_t = np.clip(end_t, 1, fdims)
                if obstruction[end_t[0] - 1, end_t[1] - 1, end_t[2] - 1] == 0:
                    ends_t[i] = end_t
                    break

        ends_tc = np.ceil(ends_t / dims_ss).astype(int)
        ends = ends_tc
        nends = ends.shape[0]

        paths = np.zeros(dims, dtype=bool)
        for i in range(nends):
            path, _ = get_dendrite_path2(pathfrom, ends[i], root)
            if path.size:
                paths[path[:, 0] - 1, path[:, 1] - 1, path[:, 2] - 1] = True

        finepaths_idx = paths.astype(np.float32) * (j + 1)
        finepaths_val = paths.astype(np.float32) * thickness_scale * dt_params[3]
        finepaths_ad = paths.astype(bool)

        fine_idxs3 = np.flatnonzero(paths)
        if fine_idxs3.size:
            xi, yi, zi = np.unravel_index(fine_idxs3, dims, order="F")
            xi = xi + allroots[j, 0] - fdims[0] // 2
            yi = yi + allroots[j, 1] - fdims[1] // 2
            zi = zi + allroots[j, 2] - fdims[2] // 2
            gidxs = (xi >= 1) & (yi >= 1) & (zi >= 1) & (xi <= fulldims[0]) & (yi <= fulldims[1]) & (zi <= fulldims[2])
            xi = xi[gidxs]
            yi = yi[gidxs]
            zi = zi[gidxs]
            didxs = np.ravel_multi_index((xi - 1, yi - 1, zi - 1), fulldims, order="F")
            fidxs = fine_idxs3[gidxs]
            _set_lin(cell_volume, didxs + 1, _get_lin(cell_volume, didxs + 1) + finepaths_idx.flat[fidxs])
            _set_lin(cell_volume_idx, didxs + 1, _get_lin(cell_volume_idx, didxs + 1) + finepaths_idx.flat[fidxs])
            _set_lin(cell_volume_val, didxs + 1, _get_lin(cell_volume_val, didxs + 1) + finepaths_val.flat[fidxs])
            _set_lin(cell_volume_ad, didxs + 1, _get_lin(cell_volume_ad, didxs + 1) | finepaths_ad.flat[fidxs])

        if vol_params.get("verbose", 1) == 1:
            print(".", end="")
        elif vol_params.get("verbose", 1) > 1:
            print(f"{j + 1} ({time.time() - start_t:.2f} seconds).")

    cell_volume_val = np.floor(cell_volume_val) + (cell_volume_val % 1 > rng.random(cell_volume_val.shape))
    cell_volume_idx = cell_volume_idx.astype(np.uint16)
    cell_volume_ad = cell_volume_ad.astype(np.uint16)
    cell_volume_bd = (~cell_volume_ad.astype(bool)).astype(np.uint16)

    _, dendnum_ad = dilate_dendrite_path_all(cell_volume_val * cell_volume_ad, cell_volume_idx * cell_volume_ad, neur_num)
    _, dendnum_bd = dilate_dendrite_path_all(cell_volume_val * cell_volume_bd, cell_volume_idx * cell_volume_bd, neur_num)
    for kk in range(n_neur):
        if kk < len(gp_nuc) and gp_nuc[kk][0] is not None:
            _set_lin(dendnum_ad, np.array(gp_nuc[kk][0], dtype=int), 0)
            _set_lin(dendnum_bd, np.array(gp_nuc[kk][0], dtype=int), 0)
        if kk < len(gp_soma) and gp_soma[kk][0] is not None:
            _set_lin(dendnum_ad, np.array(gp_soma[kk][0], dtype=int), 0)
            _set_lin(dendnum_bd, np.array(gp_soma[kk][0], dtype=int), 0)

    dendnum_bd[dendnum_ad > 0] = dendnum_ad[dendnum_ad > 0]
    neur_num[dendnum_bd > 0] = dendnum_bd[dendnum_bd > 0]

    for kk in range(n_neur):
        if kk < len(gp_nuc) and gp_nuc[kk][0] is not None:
            _set_lin(neur_num, np.array(gp_nuc[kk][0], dtype=int), 0)
        if kk < len(gp_soma) and gp_soma[kk][0] is not None:
            _set_lin(neur_num, np.array(gp_soma[kk][0], dtype=int), kk + 1)

    if vol_params.get("verbose", 1) >= 1:
        print("done.")

    return neur_num, dendnum_ad, dend_params, gp_soma


def grow_apical_dendrites(
    vol_params: Dict[str, Any],
    dend_params: Dict[str, Any],
    neur_num: np.ndarray,
    cell_volume_ad: np.ndarray,
    gp_nuc: list,
    gp_soma: list,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    vol_params = check_vol_params(vol_params)
    dend_params = check_dend_params(dend_params)
    rng = np.random.default_rng(int(dend_params.get("seed", 0)) + 3)

    vres = int(vol_params["vres"])
    vol_sz = np.array(vol_params["vol_sz"], dtype=int)
    n_den = int(vol_params["N_den"])
    n_neur = int(vol_params["N_neur"])

    if vol_params.get("verbose", 1) == 1:
        print("Growing out apical dendrites", end="")
    elif vol_params.get("verbose", 1) > 1:
        print("Growing out apical dendrites...")

    at_params = list(dend_params["atParams2"])
    dweight = float(dend_params["dweight"])
    bweight = float(dend_params["bweight"])
    thickness_scale = float(dend_params["thicknessScale"])
    dims = np.array(dend_params["dims"], dtype=int)
    dims_ss = np.array(dend_params["dimsSS"], dtype=int)
    rallexp = float(dend_params["rallexp"])

    dims = np.minimum(dims, (vol_sz / dims_ss).astype(int))
    fulldims = vol_sz * vres
    dims = dims * vres
    at_params[1] *= vres
    at_params[2] *= vres
    at_params[3] *= vres
    thickness_scale = thickness_scale * vres * vres

    fdims = np.minimum(dims * dims_ss, fulldims)

    cell_volume = neur_num.astype(np.float32)
    for kk in range(n_neur):
        if kk < len(gp_nuc) and gp_nuc[kk][0] is not None:
            _set_lin(cell_volume, np.array(gp_nuc[kk][0], dtype=int), kk + 1)

    root_den = np.zeros((n_den, 3), dtype=int)
    for j in range(n_den):
        while root_den[j, 2] == 0:
            root = np.array(
                [
                    int(rng.integers(1, fulldims[0] + 1)),
                    int(rng.integers(1, fulldims[1] + 1)),
                ],
                dtype=int,
            )
            if cell_volume[root[0] - 1, root[1] - 1, 0] == 0:
                root_den[j, :] = [root[0], root[1], fulldims[2]]

    cell_volume_idx = np.zeros_like(neur_num, dtype=np.float32)
    cell_volume_val = np.zeros_like(neur_num, dtype=np.float32)
    ML = np.full((*fdims, 6), np.inf, dtype=np.float32)

    if dend_params.get("apicalVar") is not None:
        dend_var = float(dend_params.get("apicalVar"))
    elif dend_params.get("dendVar") is not None:
        dend_var = float(dend_params.get("dendVar"))
    else:
        dend_var = 0.35

    for j in range(n_den):
        if vol_params.get("verbose", 1) > 1:
            start_t = time.time()

        rootL = np.array([fdims[0] / 2 + 1, fdims[1] / 2 + 1, root_den[j, 2]])
        borderflag = False
        Xlims = None
        Ylims = None
        try:
            obstruction = cell_volume[
                root_den[j, 0] - fdims[0] // 2 - 1 : root_den[j, 0] + fdims[0] // 2,
                root_den[j, 1] - fdims[1] // 2 - 1 : root_den[j, 1] + fdims[1] // 2,
                :,
            ]
            if obstruction.shape != tuple(fdims):
                raise ValueError("obstruction shape mismatch")
        except Exception:
            obstruction = np.zeros(fdims, dtype=np.float32)
            Xlims = np.array([root_den[j, 0] - fdims[0] // 2, root_den[j, 0] + fdims[0] // 2 - 1], dtype=int)
            Ylims = np.array([root_den[j, 1] - fdims[1] // 2, root_den[j, 1] + fdims[1] // 2 - 1], dtype=int)
            xs = slice(max(1, Xlims[0]) - 1, min(Xlims[1], fulldims[0]))
            ys = slice(max(1, Ylims[0]) - 1, min(Ylims[1], fulldims[1]))
            xdst = slice(max(1, -Xlims[0] + 2) - 1, fdims[0] + min(0, -Xlims[1] + fulldims[0]))
            ydst = slice(max(1, -Ylims[0] + 2) - 1, fdims[1] + min(0, -Ylims[1] + fulldims[1]))
            obstruction[xdst, ydst, :] = cell_volume[xs, ys, :]
            borderflag = True

        cell_body = obstruction == (j + n_neur)
        obstruction[cell_body] = 0

        root = np.ceil(rootL / dims_ss).astype(int)
        M = 1 + dweight * rng.random((*dims, 6), dtype=np.float32)
        M[0, :, :, 0] = np.inf
        M[-1, :, :, 1] = np.inf
        M[:, 0, :, 2] = np.inf
        M[:, -1, :, 3] = np.inf
        M[:, :, 0, 4] = np.inf
        M[:, :, -1, 5] = np.inf

        fillfrac = obstruction > 0
        fillfrac = fillfrac.reshape(dims_ss[0], dims[0], dims_ss[1], dims[1], dims_ss[2], dims[2])
        fillfrac = np.sum(fillfrac, axis=(0, 2, 4)) / np.prod(dims_ss)
        M = M + (-bweight * np.log(1 - (2 * np.maximum(0, fillfrac - 0.5))))[:, :, :, None]

        pathfrom = dendrite_dijkstra2(M.reshape(-1, 6, order="F"), tuple(dims), tuple(root))[1]

        endsA = np.zeros((int(at_params[0]), 3), dtype=int)
        rootA = np.floor(
            np.array(
                [
                    rootL[0] + 2 * at_params[3] * (rng.random() - 0.5),
                    rootL[1] + 2 * at_params[3] * (rng.random() - 0.5),
                    fdims[2],
                ]
            )
        ).astype(int)
        for i in range(int(at_params[0])):
            flag = True
            dist_sc = 1.0
            numit = 0
            while flag and numit < 100:
                theta = rng.random() * 2 * np.pi
                r = np.sqrt(rng.random()) * at_params[1] * dist_sc
                endA = np.floor(
                    np.array([r * np.cos(theta) + rootA[0], r * np.sin(theta) + rootA[1], 1])
                ).astype(int)
                endA[2] = fdims[2]
                endA = np.clip(endA, 1, fdims)
                if obstruction[endA[0] - 1, endA[1] - 1, endA[2] - 1] == 0:
                    endsA[i, :] = endA
                    flag = False
                dist_sc = dist_sc * 1.01
                numit += 1

        endsAC = np.ceil(endsA / dims_ss).astype(int)
        paths = np.zeros(dims, dtype=bool)
        for i in range(endsAC.shape[0]):
            _, pathM = get_dendrite_path2(pathfrom, endsAC[i, :], root)
            paths |= pathM

        ML.fill(np.inf)
        rootL = np.round((root - np.array([0.5, 0.5, 0.5])) * dims_ss).astype(int)
        den_locs = np.flatnonzero(paths.ravel(order="F"))
        for idx in den_locs:
            lx, ly, lz = np.unravel_index(idx, dims, order="F")
            temp = 1 + dweight * rng.random((*dims_ss, 6), dtype=np.float32)
            ML[
                lx * dims_ss[0] : (lx + 1) * dims_ss[0],
                ly * dims_ss[1] : (ly + 1) * dims_ss[1],
                lz * dims_ss[2] : (lz + 1) * dims_ss[2],
                :,
            ] = temp

        ML[0, :, :, 0] = np.inf
        ML[-1, :, :, 1] = np.inf
        ML[:, 0, :, 2] = np.inf
        ML[:, -1, :, 3] = np.inf
        ML[:, :, 0, 4] = np.inf
        ML[:, :, -1, 5] = np.inf
        filled = obstruction * np.inf
        filled[np.isnan(filled)] = 0
        ML = ML + filled[:, :, :, None]
        pathfromL = dendrite_dijkstra2(ML.reshape(-1, 6, order="F"), tuple(fdims), tuple(rootL))[1]

        finepaths_val = np.zeros(fdims, dtype=np.float32)
        for i in range(endsA.shape[0]):
            path = get_dendrite_path2(pathfromL, endsA[i, :], rootL)[0]
            if path.size:
                dend_sz = max(0.0, rng.normal(1.0, dend_var))
                if path.shape[0] > 2:
                    diffs = np.abs(np.diff(np.abs(np.diff(path, axis=0)), axis=0))
                    curvature = np.sum(diffs, axis=1) / 2
                    path_w = dend_sz * (1 - (1 - 1 / np.sqrt(2)) * np.concatenate([[0], curvature, [0]]))
                else:
                    path_w = dend_sz * np.ones(path.shape[0], dtype=np.float32)
                finepaths_val[path[:, 0] - 1, path[:, 1] - 1, path[:, 2] - 1] += path_w

        mask = finepaths_val > 0
        finepaths_val[mask] = thickness_scale * at_params[4] * (finepaths_val[mask] ** (1 / rallexp))
        finepaths_val[cell_body] = 0
        finepaths_idx = (j + n_neur) * mask.astype(np.float32)

        if not borderflag:
            xs = slice(root_den[j, 0] - fdims[0] // 2 - 1, root_den[j, 0] + fdims[0] // 2)
            ys = slice(root_den[j, 1] - fdims[1] // 2 - 1, root_den[j, 1] + fdims[1] // 2)
            cell_volume[xs, ys, :] += finepaths_idx
            cell_volume_val[xs, ys, :] += finepaths_val
            cell_volume_idx[xs, ys, :] += finepaths_idx
        else:
            xs = slice(max(1, Xlims[0]) - 1, min(Xlims[1], fulldims[0]))
            ys = slice(max(1, Ylims[0]) - 1, min(Ylims[1], fulldims[1]))
            xdst = slice(max(1, -Xlims[0] + 2) - 1, fdims[0] + min(0, -Xlims[1] + fulldims[0]))
            ydst = slice(max(1, -Ylims[0] + 2) - 1, fdims[1] + min(0, -Ylims[1] + fulldims[1]))
            cell_volume[xs, ys, :] += finepaths_idx[xdst, ydst, :]
            cell_volume_val[xs, ys, :] += finepaths_val[xdst, ydst, :]
            cell_volume_idx[xs, ys, :] += finepaths_idx[xdst, ydst, :]

        if vol_params.get("verbose", 1) == 1:
            print(".", end="")
        elif vol_params.get("verbose", 1) > 1:
            print(f"{j + 1} ({time.time() - start_t:.2f} seconds).")

    cell_volume_val = np.ceil(cell_volume_val).astype(np.float32)
    cell_volume_idx = cell_volume_idx.astype(np.float32)
    _, dendnum = dilate_dendrite_path_all(cell_volume_val, cell_volume_idx, neur_num)

    cell_volume_ad = cell_volume_ad.astype(np.uint16) + dendnum.astype(np.uint16)
    neur_num = neur_num.astype(np.uint16) + dendnum.astype(np.uint16)

    for kk in range(n_neur):
        if kk < len(gp_nuc) and gp_nuc[kk][0] is not None:
            _set_lin(neur_num, np.array(gp_nuc[kk][0], dtype=int), 0)
        if kk < len(gp_soma) and gp_soma[kk][0] is not None:
            _set_lin(neur_num, np.array(gp_soma[kk][0], dtype=int), kk + 1)

    neur_num_ad = cell_volume_ad.astype(np.uint16)
    for kk in range(n_neur):
        if kk < len(gp_nuc) and gp_nuc[kk][0] is not None:
            _set_lin(neur_num_ad, np.array(gp_nuc[kk][0], dtype=int), 0)
        if kk < len(gp_soma) and gp_soma[kk][0] is not None:
            _set_lin(neur_num_ad, np.array(gp_soma[kk][0], dtype=int), 0)
    neur_num_ad[(neur_num_ad - neur_num) > 0] = 0

    if vol_params.get("verbose", 1) >= 1:
        print("done.")

    return neur_num, neur_num_ad, dend_params


def set_cell_fluorescence(
    vol_params: Dict[str, Any],
    neur_params: Dict[str, Any],
    dend_params: Dict[str, Any],
    neur_num: np.ndarray,
    neur_soma: np.ndarray,
    neur_num_AD: np.ndarray,
    neur_locs: np.ndarray,
    neur_vol: np.ndarray,
) -> Tuple[list, np.ndarray]:
    """Translate setCellFluoresence.m to Python."""

    vol_params = check_vol_params(vol_params)
    neur_params = check_neur_params(neur_params)
    dend_params = check_dend_params(dend_params)

    if vol_params.get("verbose", 1) == 1:
        print("Setting Fluorescence Distribution.", end="")
    elif vol_params.get("verbose", 1) > 1:
        print("Setting Fluorescence Distribution...")

    vol_sz = np.array(vol_params["vol_sz"], dtype=int)
    n_neur = int(vol_params["N_neur"])
    vres = int(vol_params["vres"])
    gp_vals = [[None, None, None] for _ in range(n_neur + int(vol_params["N_den"]))]
    wt_sc = dend_params["weightScale"]
    fl_sc = neur_params["fluor_dist"]

    numcomps = n_neur + int(vol_params["N_den"])
    numvox = np.zeros(numcomps, dtype=int)
    currvox = np.zeros(numcomps, dtype=int)
    flat_neur = neur_num.ravel(order="F")
    for kk in range(flat_neur.size):
        val = int(flat_neur[kk])
        if 1 <= val <= numcomps:
            numvox[val - 1] += 1
    for kk in range(numcomps):
        gp_vals[kk][0] = np.zeros(numvox[kk], dtype=np.int32)
    for kk in range(flat_neur.size):
        val = int(flat_neur[kk])
        if 1 <= val <= numcomps:
            currvox[val - 1] += 1
            gp_vals[val - 1][0][currvox[val - 1] - 1] = kk + 1

    for kk in range(n_neur):
        tmp_loc = gp_vals[kk][0]
        if tmp_loc.size == 0:
            continue
        neur_soma_flat = neur_soma.ravel(order="F")
        neur_ad_flat = neur_num_AD.ravel(order="F")
        soma_mask = neur_soma_flat[tmp_loc - 1] == (kk + 1)
        ad_mask = neur_ad_flat[tmp_loc - 1] == (kk + 1)
        tmp_soma = tmp_loc[soma_mask]
        tmp_ad = tmp_loc[ad_mask]
        tmp_idxs = np.flatnonzero(soma_mask)
        tmp_idxs_ad = np.flatnonzero(ad_mask)

        tmp = masked_3dgp_v2(
            np.round(neur_params["avg_rad"] * 6 * vres).astype(int),
            fl_sc[0] * vres,
            fl_sc[1],
            0.0,
        )

        if tmp_soma.size:
            lx, ly, lz = np.unravel_index(tmp_soma - 1, vol_sz * vres, order="F")
            lx = lx + 1
            ly = ly + 1
            lz = lz + 1
            tmp_dist = np.stack([lx, ly, lz], axis=1) - np.floor(vres * neur_locs[kk, :]).astype(int)
            tmp_dist = tmp_dist + (tmp.shape[0] // 2)
            tmp_dist = np.maximum(tmp_dist, 1)
            tmp_dist = np.minimum(tmp_dist, np.array(tmp.shape, dtype=int))
            tmp_vals = tmp[tmp_dist[:, 0] - 1, tmp_dist[:, 1] - 1, tmp_dist[:, 2] - 1]
            tmp_vals = tmp_vals.astype(np.float32)
            tmp_mean = float(np.mean(tmp_vals))
            tmp_centered = tmp_vals - tmp_mean
            tmp_norm = np.max(np.abs(tmp_centered))
            if tmp_norm > 0:
                tmp_vals = 0.5 * (tmp_centered / tmp_norm) + 1.0
            else:
                tmp_vals = np.ones_like(tmp_vals, dtype=np.float32)
            tmp_vals[np.isnan(tmp_vals)] = 1.0
        else:
            tmp_vals = np.zeros(0, dtype=np.float32)

        rx, ry, rz = np.unravel_index(tmp_loc - 1, vol_sz * vres, order="F")
        rx = rx + 1
        ry = ry + 1
        rz = rz + 1
        tmp_sep = np.sqrt(
            (rx - vres * neur_locs[kk, 0]) ** 2
            + (ry - vres * neur_locs[kk, 1]) ** 2
            + (rz - vres * neur_locs[kk, 2]) ** 2
        )

        base_vals = (wt_sc[1] * np.exp(-tmp_sep / (vres * wt_sc[0])) + (1 - wt_sc[1])) * (1 - wt_sc[2])
        gp_vals[kk][1] = base_vals.astype(np.float32)
        if tmp_idxs.size:
            gp_vals[kk][1][tmp_idxs] = tmp_vals
        if tmp_idxs_ad.size:
            gp_vals[kk][1][tmp_idxs_ad] = 1.0
        gp_vals[kk][2] = np.zeros(tmp_loc.size, dtype=bool)
        if tmp_idxs.size:
            gp_vals[kk][2][tmp_idxs] = True
        if neur_vol is not None and neur_vol.size:
            _set_lin(neur_vol, tmp_loc, gp_vals[kk][1])
        if vol_params.get("verbose", 1) == 1:
            print(".", end="")

    for kk in range(n_neur, n_neur + int(vol_params["N_den"])):
        gp_vals[kk][1] = np.ones_like(gp_vals[kk][0], dtype=np.float32)
        gp_vals[kk][2] = np.zeros_like(gp_vals[kk][0], dtype=bool)
        if neur_vol is not None and neur_vol.size:
            _set_lin(neur_vol, gp_vals[kk][0], gp_vals[kk][1])

    if vol_params.get("verbose", 1) >= 1:
        print("done.")

    return gp_vals, neur_vol


def generate_bg_dendrites(
    vol_params: Dict[str, Any],
    bg_params: Dict[str, Any],
    dend_params: Dict[str, Any],
    neur_vol: np.ndarray,
    neur_num: np.ndarray,
    gp_vals: list,
    gp_nuc: list,
    neur_locs: np.ndarray | None,
    neur_vol_flag: bool | None = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], list, np.ndarray]:
    vol_params = check_vol_params(vol_params)
    dend_params = check_dend_params(dend_params)
    bg_params = check_bg_params(bg_params)
    if neur_vol_flag is None:
        neur_vol_flag = True
    if neur_locs is None:
        neur_locs = np.zeros((0, 3), dtype=np.float32)

    if vol_params.get("verbose", 1) == 1:
        print("Generating background fluorescence.", end="")
    elif vol_params.get("verbose", 1) > 1:
        print("Generating background fluorescence...")

    bg_pix = neur_num == 0
    for i in range(int(vol_params.get("N_neur", 0))):
        if i < len(gp_nuc) and isinstance(gp_nuc[i], (list, tuple)) and len(gp_nuc[i]) >= 1:
            _set_lin(bg_pix, np.array(gp_nuc[i][0], dtype=int), False)

    vres = int(vol_params["vres"])
    dt_params = list(dend_params["dtParams"])
    thickness_scale = float(dend_params["thicknessScale"])
    dt_params[1] = dt_params[1] * vres
    dt_params[2] = dt_params[2] * vres
    thickness_scale = thickness_scale * vres * vres
    volsize = np.array(vol_params["vol_sz"], dtype=int) * vres

    if vol_params.get("verbose", 1) > 1:
        print("Initializing volume", end="")
    if neur_vol_flag:
        neur_vol = np.zeros_like(neur_vol, dtype=np.float32)
        for i, entry in enumerate(gp_vals):
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                _set_lin(neur_vol, np.array(entry[0], dtype=int), entry[1])
            if i < len(gp_nuc) and isinstance(gp_nuc[i], (list, tuple)) and len(gp_nuc[i]) >= 2:
                _set_lin(neur_vol, np.array(gp_nuc[i][0], dtype=int), gp_nuc[i][1])
            if vol_params.get("verbose", 1) >= 1:
                print(".", end="")
    if vol_params.get("verbose", 1) > 1:
        print()

    rng = np.random.default_rng(int(bg_params.get("seed", 0)) + 5)
    mat = np.asfortranarray(rng.random(volsize, dtype=np.float32))
    mat[bg_pix == 0] = np.finfo(np.float32).max
    mat[0, :, :] = np.finfo(np.float32).max
    mat[:, 0, :] = np.finfo(np.float32).max
    mat[:, :, 0] = np.finfo(np.float32).max
    mat[-1, :, :] = np.finfo(np.float32).max
    mat[:, -1, :] = np.finfo(np.float32).max
    mat[:, :, -1] = np.finfo(np.float32).max

    dend_var = float(dend_params.get("dendVar", 0.25))
    idxvol = np.zeros(volsize, dtype=np.uint16)
    numvol = np.zeros(volsize, dtype=np.float32)

    maxlength = int(bg_params["maxlength"])
    distsc = float(bg_params["distsc"])
    fillweight = float(bg_params["fillweight"])
    maxel = int(bg_params["maxel"])
    minlength = int(bg_params["minlength"])
    dt_size = np.array([dt_params[1], dt_params[1], dt_params[2]], dtype=int)
    idx = 0
    shiftdist = 3

    total_loops = int(((np.prod(volsize + 2 * dt_size) / np.prod(volsize)) - 1) * vol_params["N_neur"])
    for _ in range(max(0, total_loops)):
        dendpts: list[int] = []
        root = np.floor(rng.random(3) * (volsize + 2 * dt_size) - dt_size).astype(int)
        while (root > 0).all() and (root <= volsize).all():
            root = np.floor(rng.random(3) * (volsize + 2 * dt_size) - dt_size).astype(int)
        neur_locs = np.vstack([neur_locs, (root / vres).astype(np.float32)])

        for _ in range(int(dt_params[0])):
            theta = rng.random() * 2 * np.pi
            r = np.sqrt(rng.random()) * dt_params[1]
            dends = np.floor(
                np.array(
                    [
                        r * np.cos(theta) + root[0],
                        r * np.sin(theta) + root[1],
                        2 * dt_params[2] * (rng.random() - 0.5) + root[2],
                    ]
                )
            ).astype(int)
            if (dends > 0).all() and (dends <= volsize).all():
                denom = np.where(dends - root == 0, 1, dends - root).astype(float)
                low = (root < 1) * (1 - root) / denom
                high = (root > volsize) * (volsize - root) / denom
                candidates = np.concatenate([low, high])
                shift_loc = int(np.argmax(candidates)) + 1
                max_shift = candidates[shift_loc - 1]
                bgpts = []
                numit = 0
                while not bgpts and numit < 30:
                    numit += 1
                    root2 = np.round(max_shift * (dends - root) + root).astype(int)
                    if shift_loc in (1, 4):
                        root2 += np.array([0, rng.integers(1, shiftdist + 1), rng.integers(1, shiftdist + 1)])
                    elif shift_loc in (2, 5):
                        root2 += np.array([rng.integers(1, shiftdist + 1), 0, rng.integers(1, shiftdist + 1)])
                    elif shift_loc in (3, 6):
                        root2 += np.array([rng.integers(1, shiftdist + 1), rng.integers(1, shiftdist + 1), 0])

                    root2 = np.clip(root2, 1, volsize)
                    bgpts = dendrite_randomwalk2(
                        mat,
                        tuple(root2.tolist()),
                        np.array([dends], dtype=int),
                        distsc,
                        maxlength,
                        fillweight,
                        maxel,
                        minlength,
                    ).tolist()
                    if bgpts:
                        bgpts = np.vstack([root2, np.array(bgpts, dtype=int)])
                        dend_sz = max(0.0, rng.normal(1.0, dend_var)) ** 2
                        if bgpts.shape[0] > 2:
                            diffs = np.abs(np.diff(np.abs(np.diff(bgpts, axis=0)), axis=0))
                            curvature = np.sum(diffs, axis=1) / 2
                            bgpts_w = dend_sz * (
                                1 - (1 - 1 / np.sqrt(2)) * np.concatenate([[0], curvature, [0]])
                            )
                        else:
                            bgpts_w = dend_sz * np.ones(bgpts.shape[0])
                        bgpts_i = (
                            bgpts[:, 0]
                            + (bgpts[:, 1] - 1) * volsize[0]
                            + (bgpts[:, 2] - 1) * volsize[0] * volsize[1]
                        ).astype(int)
                        dendpts.extend(bgpts_i.tolist())
                        _set_lin(numvol, bgpts_i, bgpts_w)

        if dendpts:
            idx += 1
            _set_lin(idxvol, np.array(dendpts), idx)
            current_vals = _get_lin(numvol, np.array(dendpts))
            _set_lin(numvol, np.array(dendpts), current_vals * thickness_scale * dt_params[3])

    _, pathnum = dilate_dendrite_path_all(numvol, idxvol, bg_pix == 0)
    vol_params["N_den2"] = idx
    ncomps = int(vol_params["N_neur"] + vol_params["N_den"])
    pathnum[pathnum > 0] = pathnum[pathnum > 0] + ncomps
    neur_num = neur_num + pathnum
    wt_sc = dend_params["weightScale"]

    needed = ncomps + idx
    if len(gp_vals) < needed:
        gp_vals.extend([None] * (needed - len(gp_vals)))
    for i in range(ncomps + 1, ncomps + idx + 1):
        mask = neur_num == i
        indices = np.flatnonzero(mask) + 1
        values = (
            (wt_sc[1] * np.exp(-((dt_params[1] / vres) / wt_sc[0])) + (1 - wt_sc[1]))
            * (1 - wt_sc[2] * rng.random(indices.size))
        )
        gp_vals[i - 1] = [indices.astype(np.int32), values.astype(np.float32), np.zeros(indices.size, dtype=bool)]
        if neur_vol_flag:
            _set_lin(neur_vol, indices, values.astype(np.float32))

    if not neur_vol_flag:
        neur_vol = np.array([])
    if vol_params.get("verbose", 1) >= 1:
        print("done.")
    return neur_num, neur_vol, vol_params, gp_vals, neur_locs


def generate_bgdendrites(
    vol_params: Dict[str, Any],
    bg_params: Dict[str, Any],
    dend_params: Dict[str, Any],
    neur_vol: np.ndarray,
    neur_num: np.ndarray,
    gp_vals: list,
    gp_nuc: list,
    neur_locs: np.ndarray,
    neur_vol_flag: bool | None = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], list, np.ndarray]:
    return generate_bg_dendrites(
        vol_params,
        bg_params,
        dend_params,
        neur_vol,
        neur_num,
        gp_vals,
        gp_nuc,
        neur_locs,
        neur_vol_flag,
    )


def generate_axons(
    vol_params: Dict[str, Any],
    axon_params: Dict[str, Any],
    neur_vol: np.ndarray,
    neur_num: np.ndarray,
    gp_vals: Any,
    gp_nuc: Any,
    neur_vol_flag: bool | None = True,
) -> Tuple[np.ndarray, list, Dict[str, Any], Dict[str, Any]]:
    vol_params = check_vol_params(vol_params)
    axon_params = check_axon_params(axon_params)
    if neur_vol_flag is None:
        neur_vol_flag = True

    if vol_params.get("verbose", 1) == 1:
        print("Generating background fluorescence.", end="")
    elif vol_params.get("verbose", 1) > 1:
        print("Generating background fluorescence...")

    bg_pix = neur_num == 0
    if isinstance(gp_nuc, list):
        for nuc in gp_nuc:
            if isinstance(nuc, (list, tuple)) and len(nuc) >= 1:
                _set_lin(bg_pix, np.array(nuc[0], dtype=int), False)

    fillnum = int(round(axon_params["maxfill"] * axon_params["maxvoxel"] * bg_pix.sum()))
    volsize = np.array(vol_params["vol_sz"], dtype=int) * int(vol_params["vres"])
    n_bg = int(vol_params.get("N_bg", 1))
    gp_bgvals: list = [None] * n_bg

    if vol_params.get("verbose", 1) > 1:
        print("Initializing volume")
    if neur_vol_flag:
        neur_vol = np.zeros_like(neur_vol, dtype=np.float32)
        if isinstance(gp_vals, list):
            for idx, entry in enumerate(gp_vals):
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    _set_lin(neur_vol, np.array(entry[0], dtype=int), entry[1])
                if idx < len(gp_nuc) and isinstance(gp_nuc[idx], (list, tuple)) and len(gp_nuc[idx]) >= 2:
                    _set_lin(neur_vol, np.array(gp_nuc[idx][0], dtype=int), gp_nuc[idx][1])
                if vol_params.get("verbose", 1) >= 1:
                    print(".", end="")
    if vol_params.get("verbose", 1) > 1:
        print()

    padsize = int(axon_params["padsize"])
    volpad = volsize + 2 * padsize
    rng = np.random.default_rng(int(axon_params.get("seed", 0)) + 6)

    mat = np.asfortranarray(rng.random(volpad, dtype=np.float32))
    pad_bg = np.pad(bg_pix == 0, ((padsize, padsize), (padsize, padsize), (padsize, padsize)), mode="constant", constant_values=False)
    mat[pad_bg] = np.finfo(np.float32).max

    if vol_params.get("verbose", 1) > 1:
        start_time = time.time()

    j = 1
    nummax = 10000
    while fillnum > 0 and j <= n_bg:
        bgpts = []
        numit2 = 0
        while len(bgpts) < axon_params["minlength"] and numit2 < nummax:
            numit2 += 1
            root = np.ceil((volpad - 2) * rng.random(3) + 1).astype(int)
            while mat[root[0] - 1, root[1] - 1, root[2] - 1] > (axon_params["fillweight"] * axon_params["maxvoxel"]):
                root = np.ceil((volpad - 2) * rng.random(3) + 1).astype(int)
            ends = np.ceil(root + 2 * axon_params["maxdist"] * vol_params["vres"] * (rng.random(3) - 0.5)).astype(int)
            ends = np.clip(ends, 1, volpad)
            bgpts = dendrite_randomwalk2(
                mat,
                tuple(root.tolist()),
                np.array([ends], dtype=int),
                float(axon_params["distsc"]),
                int(axon_params["maxlength"]),
                float(axon_params["fillweight"]),
                int(axon_params["maxvoxel"]),
                int(axon_params["minlength"]),
            ).tolist()

        if bgpts:
            nbranches = max(0, int(round(axon_params["numbranches"] + axon_params["varbranches"] * rng.standard_normal())))
            for _ in range(nbranches):
                bgpts2 = []
                numit = 0
                while len(bgpts2) < axon_params["minlength"] and numit < 100:
                    numit += 1
                    root = np.array(bgpts[rng.integers(0, len(bgpts))], dtype=int)
                    while (
                        root[0] == 1
                        or root[0] == volpad[0]
                        or root[1] == 1
                        or root[1] == volpad[1]
                        or root[2] == 1
                        or root[2] == volpad[2]
                    ):
                        root = np.array(bgpts[rng.integers(0, len(bgpts))], dtype=int)
                    ends = np.ceil(root + 2 * axon_params["maxdist"] * vol_params["vres"] * (rng.random(3) - 0.5)).astype(int)
                    ends = np.clip(ends, 1, volpad)
                    bgpts2 = dendrite_randomwalk2(
                        mat,
                        tuple(root.tolist()),
                        np.array([ends], dtype=int),
                        float(axon_params["distsc"]),
                        int(axon_params["maxlength"]),
                        float(axon_params["fillweight"]),
                        int(axon_params["maxvoxel"]),
                        int(axon_params["minlength"]),
                    ).tolist()
                bgpts.extend(bgpts2)

            bgpts = np.array(bgpts, dtype=int) - padsize
            valid = (
                (bgpts[:, 0] > 0)
                & (bgpts[:, 0] <= volsize[0])
                & (bgpts[:, 1] > 0)
                & (bgpts[:, 1] <= volsize[1])
                & (bgpts[:, 2] > 0)
                & (bgpts[:, 2] <= volsize[2])
            )
            bgpts = bgpts[valid]
            if bgpts.size:
                lin_idx = bgpts[:, 0] + (bgpts[:, 1] - 1) * volsize[0] + (bgpts[:, 2] - 1) * volsize[0] * volsize[1]
                gp_bgvals[j - 1] = (
                    lin_idx.astype(int),
                    (1 / axon_params["maxel"])
                    * np.ones((bgpts.shape[0], 1), dtype=np.float32)
                    * max(0.0, 1 + axon_params["varfill"] * rng.standard_normal()),
                )
                fillnum -= bgpts.shape[0]
                if neur_vol_flag:
                    current_vals = _get_lin(neur_vol, lin_idx.astype(int))
                    _set_lin(neur_vol, lin_idx.astype(int), current_vals + gp_bgvals[j - 1][1].ravel())
                j += 1
                if vol_params.get("verbose", 1) > 1 and j % 1000 == 0:
                    print(f"{j} ({time.time() - start_time:.2f} seconds).")

    if j > n_bg:
        j = n_bg
    vol_params["N_bg"] = j
    gp_bgvals = gp_bgvals[:j]

    if not neur_vol_flag:
        neur_vol = np.array([])
    if vol_params.get("verbose", 1) >= 1:
        print("done.")

    return neur_vol, gp_bgvals, axon_params, vol_params


def sort_axons(vol_params: Dict[str, Any], axon_params: Dict[str, Any], gp_bgvals: list, cell_pos: np.ndarray) -> list:
    """Translate sort_axons.m to Python."""

    vol_params = check_vol_params(vol_params)
    axon_params = check_axon_params(axon_params)

    n_proc = int(axon_params["N_proc"])
    vol_sz = np.array(vol_params["vol_sz"], dtype=int) * int(vol_params["vres"])
    if vol_params.get("verbose", 1) > 0:
        print("Sorting axons...", end="")

    bg_proc: list = [[np.array([], dtype=np.int32), np.array([], dtype=np.float32)] for _ in range(n_proc)]
    if len(bg_proc) > int(vol_params["N_neur"] + vol_params["N_den"]):
        n_comps = int(vol_params["N_neur"] + vol_params["N_den"])
        gp_bgpos = np.zeros((len(gp_bgvals), 3), dtype=float)
        for kk in range(len(gp_bgvals)):
            if gp_bgvals[kk][0] is not None and len(gp_bgvals[kk][0]) > 0:
                subs = np.column_stack(np.unravel_index(np.array(gp_bgvals[kk][0], dtype=int) - 1, vol_sz, order="F"))
                gp_bgpos[kk, :] = np.mean(subs + 1, axis=0)

        cell_pos2 = np.asarray(cell_pos, dtype=float)[:n_comps, :]
        dist_mat = np.sqrt(
            (cell_pos2[:, 0][:, None] - gp_bgpos[:, 0][None, :]) ** 2
            + (cell_pos2[:, 1][:, None] - gp_bgpos[:, 1][None, :]) ** 2
            + (cell_pos2[:, 2][:, None] - gp_bgpos[:, 2][None, :]) ** 2
        )

        idxlist = np.zeros(n_comps, dtype=int)
        for ii in range(n_comps):
            idx = int(np.argmin(dist_mat[ii, :]))
            dist_mat[:, idx] = np.inf
            bg_proc[ii][0] = np.asarray(gp_bgvals[idx][0], dtype=np.int32)
            bg_proc[ii][1] = np.asarray(gp_bgvals[idx][1], dtype=np.float32)
            idxlist[ii] = idx

        rng = np.random.default_rng()
        for kk in range(len(gp_bgvals)):
            if kk not in idxlist:
                index = n_comps + int(np.ceil((n_proc - n_comps) * rng.random())) - 1
                bg_proc[index][0] = np.concatenate([bg_proc[index][0], np.asarray(gp_bgvals[kk][0], dtype=np.int32)])
                bg_proc[index][1] = np.concatenate([bg_proc[index][1], np.asarray(gp_bgvals[kk][1], dtype=np.float32)])
    else:
        rng = np.random.default_rng()
        for kk in range(len(gp_bgvals)):
            index = int(np.ceil(n_proc * rng.random())) - 1
            bg_proc[index][0] = np.concatenate([bg_proc[index][0], np.asarray(gp_bgvals[kk][0], dtype=np.int32)])
            bg_proc[index][1] = np.concatenate([bg_proc[index][1], np.asarray(gp_bgvals[kk][1], dtype=np.float32)])

    if vol_params.get("verbose", 1) > 0:
        print("done.")

    return bg_proc


def simulate_neural_volume(
    vol_params: Dict[str, Any] | None,
    neur_params: Dict[str, Any] | None,
    vasc_params: Dict[str, Any] | None,
    dend_params: Dict[str, Any] | None,
    bg_params: Dict[str, Any] | None,
    axon_params: Dict[str, Any] | None,
    psf_params: Dict[str, Any] | None,
    debug_opt: bool | None = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    vol_params = check_vol_params(vol_params)
    neur_params = check_neur_params(neur_params)
    vasc_params = check_vasc_params(vasc_params)
    dend_params = check_dend_params(dend_params)
    bg_params = check_bg_params(bg_params)
    axon_params = check_axon_params(axon_params)

    if debug_opt:
        vol_params["verbose"] = 2

    vasc_sz = vol_params.get("vasc_sz")
    if vasc_sz is None or (isinstance(vasc_sz, (list, tuple, np.ndarray)) and len(vasc_sz) == 0):
        vasc_sz = (
            gaussian_beam_size(psf_params, vol_params["vol_depth"] + vol_params["vol_sz"][2] / 2.0)
            + np.array(vol_params["vol_sz"], dtype=float)
            + np.array([0.0, 0.0, 1.0]) * float(vol_params["vol_depth"])
        )
        vol_params["vasc_sz"] = vasc_sz.astype(int).tolist()

    if vasc_params.get("flag", True):
        neur_ves, vasc_params, neur_ves_all = simulate_blood_vessels(vol_params, vasc_params)
    else:
        vol_sz = np.array(vol_params["vol_sz"], dtype=int)
        vol_depth = int(vol_params["vol_depth"])
        vres = int(vol_params["vres"])
        neur_ves = np.zeros(tuple((vol_sz + np.array([0, 0, vol_depth])) * vres), dtype=bool)
        neur_ves_all = np.zeros(tuple(np.array(vol_params["vasc_sz"], dtype=int) * vres), dtype=bool)

    neur_locs, vcell, vnuc, tri, rot_ang = sample_dense_neurons(neur_params, vol_params, neur_ves)
    vol_params["N_neur"] = int(vcell.shape[2])

    neur_soma, neur_vol, gp_nuc, gp_soma = generate_neural_volume(neur_params, vol_params, neur_locs, vcell, vnuc, neur_ves)

    if vol_params.get("vol_depth", 0) > 0:
        vres = int(vol_params["vres"])
        offset = int(vol_params["vol_depth"] * vres) * int(np.prod(np.array(vol_params["vol_sz"][:2]) * vres))
        soma_idx = np.flatnonzero(neur_soma.ravel(order="F") > 0)
        target = soma_idx + offset
        target = target[target < neur_ves.size]
        _set_lin(neur_ves, target + 1, False)

    neur_num, cell_volume_ad, dend_params, gp_soma = grow_neuron_dendrites(
        vol_params, dend_params, neur_soma, neur_ves, neur_locs, gp_nuc, gp_soma, rot_ang
    )
    neur_num, neur_num_ad, dend_params = grow_apical_dendrites(
        vol_params, dend_params, neur_num, cell_volume_ad, gp_nuc, gp_soma
    )
    gp_vals, neur_vol = set_cell_fluorescence(
        vol_params, neur_params, dend_params, neur_num, neur_soma, neur_num_ad, neur_locs, neur_vol
    )

    if bg_params.get("flag"):
        neur_num, neur_vol, vol_params, gp_vals, neur_locs = generate_bg_dendrites(
            vol_params, bg_params, dend_params, neur_vol, neur_num, gp_vals, gp_nuc, neur_locs
        )

    gp_bgvals: Any = []
    bg_proc: Any = []
    if axon_params.get("flag"):
        neur_vol, gp_bgvals, axon_params, _ = generate_axons(
            vol_params, axon_params, neur_vol, neur_num, gp_vals, gp_nuc
        )
        axon_params["N_proc"] = len(gp_vals)
        bg_proc = sort_axons(vol_params, axon_params, gp_bgvals, neur_locs * vol_params["vres"])

    if vol_params.get("verbose", 1) >= 1:
        print("Setting up output struct...", end="")

    vol_out: Dict[str, Any] = {
        "neur_vol": neur_vol,
        "gp_nuc": gp_nuc,
        "gp_soma": gp_soma,
        "gp_vals": gp_vals,
        "neur_ves": neur_ves,
        "bg_proc": bg_proc,
        "neur_ves_all": neur_ves_all,
        "locs": neur_locs,
    }

    if debug_opt:
        vol_out.update(
            {
                "neur_num_AD": neur_num_ad,
                "neur_soma": neur_soma,
                "neur_num": neur_num,
                "gp_bgvals": gp_bgvals,
                "Vcell": vcell,
                "Vnuc": vnuc,
                "Tri": tri,
            }
        )

    if vol_params.get("verbose", 1) >= 1:
        print("done.")

    return vol_out, vol_params, neur_params, vasc_params, dend_params, bg_params, axon_params


def branch_grow_nodes(nodes: list, neur_ves: np.ndarray, params: Dict[str, Any], idx: int, direction: float) -> Tuple[list, np.ndarray]:
    rng = params.get("_rng")
    if rng is None:
        rng = np.random.default_rng()
        params["_rng"] = rng

    border_flag = True
    overlap_flag = False
    num_iters = 0
    prev_pos = np.asarray(nodes[idx]["pos"], dtype=float)[:2]
    prev_num = idx
    pending_coords: Tuple[np.ndarray, np.ndarray] | None = None
    nv_size = neur_ves.shape
    branch_prob = 0.0
    structure = _disk_structuring_element(params.get("vesrad", 1))
    neur_ves2 = _binary_dilate(neur_ves.astype(bool), structure)

    max_iterations = int(params.get("maxit", 50))
    lensc = float(params.get("lensc", 5))
    varsc = float(params.get("varsc", 1))
    mindist = float(params.get("mindist", 1))
    varpos = float(params.get("varpos", 0.0))
    dirvar = float(params.get("dirvar", np.pi / 4))
    branchp = float(params.get("branchp", 0.1))

    while num_iters < max_iterations and border_flag:
        if rng.random() < branch_prob:
            branch_prob = 0.0
            dir_b = direction - abs((0.5 + rng.random()) * dirvar)
            direction = direction + abs((0.5 + rng.random()) * dirvar)
            nodes, neur_ves = branch_grow_nodes(nodes, neur_ves, params, prev_num, dir_b)
            neur_ves2 = _binary_dilate(neur_ves.astype(bool), structure)
            overlap_flag = True
        else:
            branch_prob += branchp

        dir_vect = np.array([np.cos(direction), np.sin(direction)])
        ves_dist = max(varsc * rng.standard_normal() + lensc, mindist)
        node_pos = dir_vect * ves_dist + prev_pos + varpos * rng.standard_normal(size=2)
        clamp_max = np.array(nv_size[:2], dtype=float)
        node_pos = np.clip(node_pos, 1.0, clamp_max)
        if np.any((node_pos <= 1) | (node_pos >= clamp_max)):
            border_flag = False

        steps = max(2, int(np.ceil(ves_dist)))
        test_subs = np.round(_vec_linspace(node_pos, prev_pos, steps)).astype(int)
        plus_y = test_subs + np.array([[0], [1]])
        plus_x = test_subs + np.array([[1], [0]])
        test_subs = np.concatenate([test_subs, plus_y, plus_x], axis=1)
        rows = np.clip(test_subs[0] - 1, 0, nv_size[0] - 1)
        cols = np.clip(test_subs[1] - 1, 0, nv_size[1] - 1)
        path_occupied = neur_ves2[rows, cols]
        if (not path_occupied.any()) or overlap_flag:
            node_num = len(nodes)
            nodes.append(generate_node(node_num, prev_num, prev_num, node_pos, "surf", {}))
            prev_pos = node_pos
            prev_num = node_num
            num_iters += 1
            if pending_coords is not None:
                neur_ves[pending_coords] = True
            pending_coords = (rows, cols)
        else:
            border_flag = False
            pending_coords = None
        overlap_flag = False

    if pending_coords is not None:
        neur_ves[pending_coords] = True
    return nodes, neur_ves


def conn_to_vol(nodes: list, conn: list, nv: Any, idxs: Any = None, neur_ves: np.ndarray | None = None) -> Tuple[np.ndarray, list]:
    if neur_ves is None:
        neur_ves = np.zeros(nv["size"], dtype=bool)
    if idxs is None:
        idxs = list(range(len(conn)))

    rng = np.random.default_rng()
    nv_size = np.array(nv["size"], dtype=int)

    for conn_idx in idxs:
        start_idx = conn[conn_idx]["start"]
        end_idx = conn[conn_idx]["ends"]
        start_pos = np.asarray(nodes[start_idx]["pos"], dtype=float)
        end_pos = np.asarray(nodes[end_idx]["pos"], dtype=float)

        left_candidates = list(set(nodes[start_idx]["conn"]) - {end_idx})
        right_candidates = list(set(nodes[end_idx]["conn"]) - {start_idx})
        left_idx = rng.choice(left_candidates) if left_candidates else None
        right_idx = rng.choice(right_candidates) if right_candidates else None

        control_points = [start_pos, end_pos]
        if left_idx is not None:
            control_points.insert(0, np.asarray(nodes[left_idx]["pos"], dtype=float))
        if right_idx is not None:
            control_points.append(np.asarray(nodes[right_idx]["pos"], dtype=float))

        seg_lengths = [np.linalg.norm(control_points[i + 1] - control_points[i]) for i in range(len(control_points) - 1)]
        total_len = max(sum(seg_lengths), 1.0)
        num_samples = max(2, int(np.ceil(2 * np.linalg.norm(start_pos - end_pos))))

        samples = []
        for i, seg_len in enumerate(seg_lengths):
            seg_samples = max(2, int(np.ceil(num_samples * (seg_len / total_len))))
            segment = np.linspace(control_points[i], control_points[i + 1], seg_samples, endpoint=False)
            samples.append(segment)
        samples.append(control_points[-1][None, :])
        ves_loc = np.ceil(np.vstack(samples)).astype(int)

        ves_loc = np.clip(ves_loc, 1, nv_size)
        ves_loc = np.unique(ves_loc, axis=0)
        conn[conn_idx]["locs"] = ves_loc

        weight = float(conn[conn_idx].get("weight", 1))
        min_idx = np.maximum(ves_loc.min(axis=0) - int(np.ceil(weight)), [1, 1, 1])
        max_idx = np.minimum(ves_loc.max(axis=0) + int(np.ceil(weight)), nv_size)
        local_loc = ves_loc - min_idx + 1

        local_shape = max_idx - min_idx + 1
        tmp = np.zeros(tuple(local_shape), dtype=bool)
        tmp[local_loc[:, 0] - 1, local_loc[:, 1] - 1, local_loc[:, 2] - 1] = True
        structure = _ball_structuring_element(int(np.ceil(weight)))
        tmp = _binary_dilate_3d(tmp, structure)

        min_idx0 = min_idx - 1
        max_idx0 = max_idx
        neur_ves[min_idx0[0]:max_idx0[0], min_idx0[1]:max_idx0[1], min_idx0[2]:max_idx0[2]] |= tmp

    return neur_ves, conn


def del_node(nodes: list, num: int) -> list:
    if not (0 <= num < len(nodes)):
        return nodes
    for idx in list(nodes[num].get("conn", [])):
        if 0 <= idx < len(nodes):
            nodes[idx]["conn"] = [c for c in nodes[idx].get("conn", []) if c != num]
            if nodes[idx].get("root") == num:
                nodes[idx]["root"] = []
    nodes[num] = generate_node(num, [], [], [0, 0, 0], "", {})
    return nodes


def dendrite_dijkstra2(m: np.ndarray, dims: Tuple[int, int, int], root: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    dims = tuple(int(v) for v in dims)
    root_idx = tuple(int(v) - 1 for v in root)
    if any(v < 0 or v >= dims[i] for i, v in enumerate(root_idx)):
        raise ValueError("root")

    weights = np.asarray(m, dtype=np.float32)
    if weights.ndim == 4 and weights.shape[-1] == 6:
        weights = weights.reshape(-1, 6, order="F")
    elif weights.ndim == 2 and weights.shape[1] == 6:
        weights = weights.reshape(-1, 6)
    else:
        raise ValueError("Unexpected M shape for dendrite_dijkstra2")

    n_nodes = weights.shape[0]
    dist = np.full(n_nodes, np.finfo(np.float32).max, dtype=np.float32)
    prev = np.full(n_nodes, -1, dtype=np.int32)

    root_linear = root_idx[0] + root_idx[1] * dims[0] + root_idx[2] * dims[0] * dims[1]
    dist[root_linear] = 0.0

    import heapq

    heap: list[tuple[float, int]] = [(0.0, root_linear)]
    to_visit = np.ones(n_nodes, dtype=bool)

    def neighbors(idx: int) -> list[tuple[int, int]]:
        x = idx % dims[0]
        y = (idx // dims[0]) % dims[1]
        z = idx // (dims[0] * dims[1])
        out = []
        if x + 1 < dims[0]:
            out.append((idx + 1, 0))
        if x - 1 >= 0:
            out.append((idx - 1, 1))
        if y + 1 < dims[1]:
            out.append((idx + dims[0], 2))
        if y - 1 >= 0:
            out.append((idx - dims[0], 3))
        if z + 1 < dims[2]:
            out.append((idx + dims[0] * dims[1], 4))
        if z - 1 >= 0:
            out.append((idx - dims[0] * dims[1], 5))
        return out

    while heap:
        current_dist, idx = heapq.heappop(heap)
        for nb_idx, weight_idx in neighbors(idx):
            if nb_idx < 0 or nb_idx >= n_nodes:
                continue
            if not to_visit[nb_idx]:
                continue
            weight = weights[nb_idx, weight_idx]
            cand = current_dist + float(weight)
            if cand < dist[nb_idx]:
                dist[nb_idx] = cand
                prev[nb_idx] = idx
                heapq.heappush(heap, (cand, nb_idx))
        to_visit[idx] = False

    distance = dist.reshape(dims, order="F")
    pathfrom = np.zeros((n_nodes, 3), dtype=np.int32)
    valid = np.where(prev >= 0)[0]
    if valid.size:
        subs = np.array(np.unravel_index(prev[valid], dims, order="F")).T + 1
        pathfrom[valid] = subs
    pathfrom = pathfrom.reshape((*dims, 3), order="F")
    return distance, pathfrom


def dendrite_randomwalk2(
    M: np.ndarray,
    root: Tuple[int, int, int],
    ends: np.ndarray,
    distsc: float,
    maxlength: int,
    fillweight: float,
    maxel: int,
    minlength: int,
) -> np.ndarray:
    mat = np.asarray(M, dtype=np.float32, order="F")
    mat_lin = mat.ravel(order="F")
    vol_size = np.array(mat.shape, dtype=int)
    root_idx = np.array(root, dtype=int) - 1
    ends_idx = np.array(ends[0], dtype=int) - 1

    curr = root_idx.copy()
    curr_idx = curr[0] + curr[1] * vol_size[0] + curr[2] * vol_size[0] * vol_size[1]
    ends_linear = ends_idx[0] + ends_idx[1] * vol_size[0] + ends_idx[2] * vol_size[0] * vol_size[1]
    maxfill = maxel * fillweight

    path = np.zeros((maxlength, 3), dtype=np.int32)
    mvals = np.zeros(maxlength, dtype=np.float32)
    bglength = 0

    for i in range(maxlength):
        dist = np.linalg.norm(ends_idx - curr) / max(distsc, np.finfo(np.float32).eps)
        distvec = (curr - ends_idx) / dist if dist != 0 else np.zeros(3, dtype=float)

        jmin = 6
        minmat = np.finfo(np.float32).max

        if curr[0] != (vol_size[0] - 1):
            cand = mat_lin[curr_idx + 1] + distvec[0]
            if cand < minmat:
                jmin = 0
                minmat = cand
        if curr[1] != (vol_size[1] - 1):
            cand = mat_lin[curr_idx + vol_size[0]] + distvec[1]
            if cand < minmat:
                jmin = 1
                minmat = cand
        if curr[2] != (vol_size[2] - 1):
            cand = mat_lin[curr_idx + vol_size[0] * vol_size[1]] + distvec[2]
            if cand < minmat:
                jmin = 2
                minmat = cand
        if curr[0] != 0:
            cand = mat_lin[curr_idx - 1] - distvec[0]
            if cand < minmat:
                jmin = 3
                minmat = cand
        if curr[1] != 0:
            cand = mat_lin[curr_idx - vol_size[0]] - distvec[1]
            if cand < minmat:
                jmin = 4
                minmat = cand
        if curr[2] != 0:
            cand = mat_lin[curr_idx - vol_size[0] * vol_size[1]] - distvec[2]
            if cand < minmat:
                jmin = 5
                minmat = cand

        if minmat < maxfill:
            if jmin == 0:
                curr[0] += 1
            elif jmin == 1:
                curr[1] += 1
            elif jmin == 2:
                curr[2] += 1
            elif jmin == 3:
                curr[0] -= 1
            elif jmin == 4:
                curr[1] -= 1
            elif jmin == 5:
                curr[2] -= 1
            curr_idx = curr[0] + curr[1] * vol_size[0] + curr[2] * vol_size[0] * vol_size[1]
            path[i] = curr
            mvals[i] = minmat
            mat_lin[curr_idx] = np.finfo(np.float32).max
            if (
                curr[0] == 0
                or curr[1] == 0
                or curr[2] == 0
                or curr[0] == (vol_size[0] - 1)
                or curr[1] == (vol_size[1] - 1)
                or curr[2] == (vol_size[2] - 1)
            ):
                break
        else:
            bglength = i - 1
            break
        if curr_idx == ends_linear:
            break

    if bglength <= 0:
        bglength = i

    if bglength >= minlength:
        for j in range(bglength):
            if mvals[j] < maxfill:
                idx = path[j, 0] + path[j, 1] * vol_size[0] + path[j, 2] * vol_size[0] * vol_size[1]
                mat_lin[idx] = mvals[j] + fillweight
    else:
        for j in range(bglength):
            idx = path[j, 0] + path[j, 1] * vol_size[0] + path[j, 2] * vol_size[0] * vol_size[1]
            mat_lin[idx] = mvals[j]

    path_out = path[:bglength] + 1
    return path_out


def dilate_dendrite_path_all(paths: np.ndarray, pathnums: np.ndarray, obstruction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    max_dist = 20
    grid = np.arange(-max_dist, max_dist + 1)
    x, y, z = np.meshgrid(grid, grid, grid, indexing="xy")
    dists = x * x + y * y + z * z
    dsz = dists.shape
    dval = np.sort(dists.ravel())
    didx = np.argsort(dists.ravel())
    dpos = np.flatnonzero(np.diff(dval))

    paths = paths.astype(np.float32, copy=True)
    paths[obstruction.astype(bool)] = np.nan
    dims = paths.shape
    pdims = int(np.prod(dims))
    dshifts = np.array([-dims[0] * dims[1], dims[0] * dims[1], -dims[0], dims[0], -1, 1], dtype=int)

    idxs = np.flatnonzero(paths > 1)
    i = 0
    while i < max_dist**2 and idxs.size:
        start = dpos[i] + 1
        end = dpos[i + 1] if i + 1 < dpos.size else didx.size
        shift_idx = didx[start:end]
        dx, dy, dz = np.unravel_index(shift_idx, dsz, order="F")
        dx = dx - max_dist
        dy = dy - max_dist
        dz = dz - max_dist
        jidxs = (dz) * dims[0] * dims[1] - (dy) * dims[0] - dx

        for idx in idxs:
            pidxs = idx + jidxs
            pidxs = pidxs[(pidxs > 0) & (pidxs < pdims)]
            pidxs = pidxs[paths.ravel()[pidxs] == 0]
            numval = pathnums.ravel()[idx]
            if pidxs.size:
                keep = []
                for pidx in pidxs:
                    didxs = pidx + dshifts
                    didxs = didxs[(didxs > 0) & (didxs < pdims)]
                    if np.any(pathnums.ravel()[didxs] == numval):
                        keep.append(pidx)
                pidxs = np.array(keep, dtype=int)
            if pidxs.size:
                while paths.ravel()[idx] > 1 and pidxs.size:
                    ridx = np.random.randint(0, pidxs.size)
                    pidx = pidxs[ridx]
                    pidxs = np.delete(pidxs, ridx)
                    paths.ravel()[idx] -= 1
                    paths.ravel()[pidx] = 1
                    pathnums.ravel()[pidx] = numval

        idxs = np.flatnonzero(paths > 1)
        i += 1

    return paths, pathnums


def get_dendrite_path2(pathfrom: np.ndarray, node: np.ndarray, root: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    node = np.asarray(node, dtype=int)
    root = np.asarray(root, dtype=int)
    dims = node.size

    if dims == 2:
        path = [node.copy()]
        while not np.array_equal(node, root):
            node = np.squeeze(pathfrom[node[0] - 1, node[1] - 1, :]).astype(int)
            path.append(node.copy())
        path = np.vstack(path)
        path_m = np.zeros(pathfrom.shape[:2], dtype=bool)
        if path.size:
            path_m[path[:, 0] - 1, path[:, 1] - 1] = True
        return path, path_m

    if dims == 3:
        path = np.zeros((np.sum(pathfrom.shape[:3]), 3), dtype=int)
        path[0, :] = node
        idx = 0
        while not np.array_equal(node, root):
            try:
                idx += 1
                node = np.reshape(pathfrom[node[0] - 1, node[1] - 1, node[2] - 1, :], (3,))
                path[idx, :] = node
            except Exception:
                return np.array([]), np.zeros(pathfrom.shape[:3], dtype=bool)
        path = path[: idx + 1]
        path_m = np.zeros(pathfrom.shape[:3], dtype=bool)
        if path.size:
            path_m[path[:, 0] - 1, path[:, 1] - 1, path[:, 2] - 1] = True
        return path, path_m

    raise ValueError("number of dimension of node is not 2 or 3")


def pseudo_rand_sample2d(sz: Tuple[int, int], nsamps: int, width: float, weight: float, pdf: np.ndarray | None, maxit: int) -> Tuple[np.ndarray, np.ndarray]:
    if maxit is None:
        maxit = 1000
    if pdf is None:
        pdf = np.ones(sz, dtype=np.float32)
    if weight is None:
        weight = 1.0
    if width is None:
        width = 2.0

    rng = np.random.default_rng()
    radius = int(np.ceil(2 * width))
    x, y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1), indexing="xy")
    gpdf = (-weight * np.exp(-(x**2 + y**2) / (width**2))).astype(np.float32)

    pos = np.zeros((nsamps, 2), dtype=np.int32)
    i = 0
    numit = 0
    while i < nsamps and numit < maxit:
        numit += 1
        rndpt = np.ceil(rng.random(2) * np.array(sz)).astype(int)
        if (pdf[rndpt[0] - 1, rndpt[1] - 1] - rng.random()) > 0:
            xc = np.array(
                [
                    max(0, 2 * width + 1 - rndpt[0]),
                    min(0, sz[0] - rndpt[0] - 2 * width) + 4 * width,
                ],
                dtype=int,
            ) + 1
            yc = np.array(
                [
                    max(0, 2 * width + 1 - rndpt[1]),
                    min(0, sz[1] - rndpt[1] - 2 * width) + 4 * width,
                ],
                dtype=int,
            ) + 1
            xi = np.array([max(1, rndpt[0] - 2 * width), min(sz[0], rndpt[0] + 2 * width)], dtype=int)
            yi = np.array([max(1, rndpt[1] - 2 * width), min(sz[1], rndpt[1] + 2 * width)], dtype=int)

            pdf[xi[0] - 1 : xi[1], yi[0] - 1 : yi[1]] += gpdf[xc[0] - 1 : xc[1], yc[0] - 1 : yc[1]]
            pos[i] = rndpt
            i += 1
            numit = 0

    return pos, pdf


def pseudo_rand_sample3d(sz: Tuple[int, int, int], nsamps: int, width: float, weight: float, pdf: np.ndarray | None) -> Tuple[np.ndarray, np.ndarray]:
    if pdf is None:
        pdf = np.ones(sz, dtype=np.float32)
    if weight is None:
        weight = 1.0
    if width is None:
        width = 2.0

    rng = np.random.default_rng()
    radius = int(np.ceil(2 * width))
    grid = np.arange(-radius, radius + 1)
    x, y, z = np.meshgrid(grid, grid, grid, indexing="xy")
    gpdf = (-weight * np.exp(-(x**2 + y**2 + z**2) / (width**2))).astype(np.float32)

    pos = np.zeros((nsamps, 3), dtype=np.int32)
    i = 0
    while i < nsamps:
        rndpt = np.ceil(rng.random(3) * np.array(sz)).astype(int)
        if (pdf[rndpt[0] - 1, rndpt[1] - 1, rndpt[2] - 1] - rng.random()) > 0:
            xc = np.array(
                [
                    max(0, 2 * width + 1 - rndpt[0]),
                    min(0, sz[0] - rndpt[0] - 2 * width) + 4 * width,
                ],
                dtype=int,
            ) + 1
            yc = np.array(
                [
                    max(0, 2 * width + 1 - rndpt[1]),
                    min(0, sz[1] - rndpt[1] - 2 * width) + 4 * width,
                ],
                dtype=int,
            ) + 1
            zc = np.array(
                [
                    max(0, 2 * width + 1 - rndpt[2]),
                    min(0, sz[2] - rndpt[2] - 2 * width) + 4 * width,
                ],
                dtype=int,
            ) + 1

            xi = np.array([max(1, rndpt[0] - 2 * width), min(sz[0], rndpt[0] + 2 * width)], dtype=int)
            yi = np.array([max(1, rndpt[1] - 2 * width), min(sz[1], rndpt[1] + 2 * width)], dtype=int)
            zi = np.array([max(1, rndpt[2] - 2 * width), min(sz[2], rndpt[2] + 2 * width)], dtype=int)

            pdf[xi[0] - 1 : xi[1], yi[0] - 1 : yi[1], zi[0] - 1 : zi[1]] += gpdf[
                xc[0] - 1 : xc[1], yc[0] - 1 : yc[1], zc[0] - 1 : zc[1]
            ]
            pos[i] = rndpt
            i += 1

    return pos, pdf


def _update_pickle(path: Path, data: Dict[str, Any]) -> None:
    existing: Dict[str, Any] = {}
    if path.exists():
        try:
            with path.open("rb") as handle:
                existing = pickle.load(handle)
        except Exception:
            existing = {}
    existing.update(data)
    with path.open("wb") as handle:
        pickle.dump(existing, handle)


def simulate_neural_volume_lowram(
    vol_params: Dict[str, Any],
    neur_params: Dict[str, Any],
    vasc_params: Dict[str, Any],
    dend_params: Dict[str, Any],
    bg_params: Dict[str, Any],
    axon_params: Dict[str, Any],
    psf_params: Dict[str, Any],
    save_loc: str | None = None,
    debug_opt: bool | None = False,
    vol_out_flag: bool | None = True,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if vol_out_flag is None:
        vol_out_flag = True

    vol_params = check_vol_params(vol_params)
    neur_params = check_neur_params(neur_params)
    vasc_params = check_vasc_params(vasc_params)
    dend_params = check_dend_params(dend_params)
    bg_params = check_bg_params(bg_params)
    axon_params = check_axon_params(axon_params)

    if debug_opt:
        vol_params["verbose"] = 2

    save_path = Path(save_loc) if save_loc else None
    if save_path and not save_path.exists():
        _update_pickle(
            save_path,
            {
                "vol_params": vol_params,
                "neur_params": neur_params,
                "vasc_params": vasc_params,
                "dend_params": dend_params,
                "bg_params": bg_params,
                "axon_params": axon_params,
                "psf_params": psf_params,
            },
        )

    vasc_sz = vol_params.get("vasc_sz")
    if vasc_sz is None or (isinstance(vasc_sz, (list, tuple, np.ndarray)) and len(vasc_sz) == 0):
        vasc_sz = (
            gaussian_beam_size(psf_params, vol_params["vol_depth"] + vol_params["vol_sz"][2] / 2.0)
            + np.array(vol_params["vol_sz"], dtype=float)
            + np.array([0.0, 0.0, 1.0]) * float(vol_params["vol_depth"])
        )
        vol_params["vasc_sz"] = vasc_sz.astype(int).tolist()

    if vasc_params.get("flag", True):
        neur_ves, vasc_params, neur_ves_all = simulate_blood_vessels(vol_params, vasc_params)
    else:
        vol_sz = np.array(vol_params["vol_sz"], dtype=int)
        vol_depth = int(vol_params["vol_depth"])
        vres = int(vol_params["vres"])
        neur_ves = np.zeros(tuple((vol_sz + np.array([0, 0, vol_depth])) * vres), dtype=bool)
        neur_ves_all = np.zeros(tuple(np.array(vol_params["vasc_sz"], dtype=int) * vres), dtype=bool)

    if save_path:
        _update_pickle(save_path, {"neur_ves_all": neur_ves_all})

    vol_out: Dict[str, Any] = {}
    vol_out["neur_ves_all"] = neur_ves_all

    neur_locs, vcell, vnuc, _, rot_ang = sample_dense_neurons(neur_params, vol_params, neur_ves)
    vol_params["N_neur"] = int(vcell.shape[2])

    neur_soma, _, gp_nuc, gp_soma = generate_neural_volume(neur_params, vol_params, neur_locs, vcell, vnuc, neur_ves)

    if vol_params.get("vol_depth", 0) > 0:
        vres = int(vol_params["vres"])
        offset = int(vol_params["vol_depth"] * vres) * int(np.prod(np.array(vol_params["vol_sz"][:2]) * vres))
        soma_idx = np.flatnonzero(neur_soma.ravel(order="F") > 0)
        target = soma_idx + offset
        target = target[target < neur_ves.size]
        _set_lin(neur_ves, target + 1, False)

    neur_num, cell_volume_ad, dend_params, gp_soma = grow_neuron_dendrites(
        vol_params, dend_params, neur_soma, neur_ves, neur_locs, gp_nuc, gp_soma, rot_ang
    )
    neur_num, neur_num_ad, dend_params = grow_apical_dendrites(
        vol_params, dend_params, neur_num, cell_volume_ad, gp_nuc, gp_soma
    )

    gp_vals, _ = set_cell_fluorescence(
        vol_params, neur_params, dend_params, neur_num, neur_soma, neur_num_ad, neur_locs, np.array([])
    )

    if bg_params.get("flag"):
        neur_num, _, vol_params, gp_vals, neur_locs = generate_bg_dendrites(
            vol_params, bg_params, dend_params, np.array([]), neur_num, gp_vals, gp_nuc, neur_locs, False
        )

    if axon_params.get("flag"):
        _, gp_bgvals, axon_params, _ = generate_axons(
            vol_params, axon_params, np.array([]), neur_num, gp_vals, gp_nuc, False
        )
        axon_params["N_proc"] = len(gp_vals)
        bg_proc = sort_axons(vol_params, axon_params, gp_bgvals, neur_locs * vol_params["vres"])
    else:
        bg_proc = []

    if save_path:
        _update_pickle(
            save_path,
            {
                "bg_proc": bg_proc,
                "gp_nuc": gp_nuc,
                "gp_vals": gp_vals,
                "neur_locs": neur_locs,
                "vol_params": vol_params,
                "neur_params": neur_params,
                "vasc_params": vasc_params,
                "dend_params": dend_params,
                "bg_params": bg_params,
                "axon_params": axon_params,
                "psf_params": psf_params,
            },
        )

    if vol_out_flag:
        vol_out["bg_proc"] = bg_proc
        vol_out["gp_nuc"] = gp_nuc
        vol_out["gp_vals"] = gp_vals
        vol_out["locs"] = neur_locs

    return vol_out, vol_params, neur_params, vasc_params, dend_params, bg_params, axon_params


def smooth_cell_body(allpaths: Any, cell_body: np.ndarray, fdims: Tuple[int, int, int]) -> np.ndarray:
    """Translate smoothCellBody.m to Python."""

    conn_idx_root = np.zeros((len(allpaths), 3), dtype=int)
    empty_idxs = np.zeros(len(allpaths), dtype=bool)
    cell_body = np.asarray(cell_body, dtype=int)
    fdims = tuple(int(v) for v in fdims)

    cell_body_set = set(cell_body.tolist())
    for i, path in enumerate(allpaths):
        if path is None or len(path) == 0:
            empty_idxs[i] = True
            continue
        path = np.asarray(path, dtype=int)
        path_ind = np.ravel_multi_index((path[:, 0] - 1, path[:, 1] - 1, path[:, 2] - 1), fdims, order="F") + 1
        path_intersect = np.isin(path_ind, cell_body)
        try:
            idx = int(np.flatnonzero(path_intersect)[0])
            conn_idx_root[i, :] = path[idx, :]
        except Exception:
            empty_idxs[i] = True

    dist_mat = np.sqrt(
        (conn_idx_root[:, 0][:, None] - conn_idx_root[:, 0][None, :]) ** 2
        + (conn_idx_root[:, 1][:, None] - conn_idx_root[:, 1][None, :]) ** 2
        + (conn_idx_root[:, 2][:, None] - conn_idx_root[:, 2][None, :]) ** 2
    )
    dist_mat = (dist_mat == 0).astype(float)
    if np.any(empty_idxs):
        dist_mat[empty_idxs, :] = np.nan

    dend_groups: list = []
    for i in range(dist_mat.shape[0]):
        if not np.isnan(dist_mat[i, i]):
            group = np.flatnonzero(dist_mat[i, :]).tolist()
            dend_groups.append(group)
            dist_mat[group, :] = np.nan

    offset = 2
    conn_idx = np.zeros((len(dend_groups), 3), dtype=int)
    conn_roots = np.zeros((len(dend_groups), 3), dtype=int)
    for i, group in enumerate(dend_groups):
        path = np.asarray(allpaths[group[0]], dtype=int)
        path_ind = np.ravel_multi_index((path[:, 0] - 1, path[:, 1] - 1, path[:, 2] - 1), fdims, order="F") + 1
        path_intersect = np.isin(path_ind, cell_body)
        try:
            idx = int(np.flatnonzero(path_intersect)[0])
            conn_idx[i, :] = path[max(0, idx - int(round(offset * np.sqrt(len(group))))) - 1, :]
            conn_roots[i, :] = path[idx, :]
        except Exception:
            conn_idx[i, :] = path[0, :]
            conn_roots[i, :] = path[0, :]

    xi, yi, zi = np.unravel_index(cell_body - 1, fdims, order="F")
    cell_ind = np.column_stack([xi + 1, yi + 1, zi + 1])
    cell_min = cell_ind.min(axis=0)
    cell_max = cell_ind.max(axis=0)

    cell_mat = np.zeros(fdims, dtype=bool)
    _set_lin(cell_mat, cell_body, True)
    cell_crop = cell_mat[
        cell_min[0] - 1 : cell_max[0],
        cell_min[1] - 1 : cell_max[1],
        cell_min[2] - 1 : cell_max[2],
    ]

    cell_diff = (
        cell_crop[:-2, 1:-1, 1:-1]
        + cell_crop[2:, 1:-1, 1:-1]
        + cell_crop[1:-1, :-2, 1:-1]
        + cell_crop[1:-1, 2:, 1:-1]
        + cell_crop[1:-1, 1:-1, :-2]
        + cell_crop[1:-1, 1:-1, 2:]
    )
    cell_borders = cell_crop.copy()
    inner = cell_borders[1:-1, 1:-1, 1:-1]
    cell_borders[1:-1, 1:-1, 1:-1] = (cell_diff > 0) & (cell_diff < 6) & inner

    cell_borders2 = np.zeros(fdims, dtype=bool)
    cell_borders2[
        cell_min[0] - 1 : cell_max[0],
        cell_min[1] - 1 : cell_max[1],
        cell_min[2] - 1 : cell_max[2],
    ] = cell_borders

    borders_sub = np.column_stack(np.nonzero(cell_borders2))
    if borders_sub.size > 0:
        borders_sub = borders_sub + 1

    cell_processed = np.zeros(fdims, dtype=bool)
    test_dist = np.array([0, 4, 10], dtype=float)
    numsamp = 20
    for j in range(conn_roots.shape[0]):
        dist_off = min(max(test_dist[1], round(offset * np.sqrt(len(dend_groups[j])))), test_dist[2])
        border_dist = conn_roots[j, :] - borders_sub
        border_dist = np.sqrt(np.sum(border_dist**2, axis=1))
        test_idx = np.where((border_dist < dist_off) & (border_dist > test_dist[0]))[0]

        test_sub = []
        for idx in test_idx:
            p0 = conn_roots[j, :]
            p1 = conn_idx[j, :]
            p2 = borders_sub[idx, :]
            sample = np.vstack(
                [
                    np.linspace(p0[0], p1[0], numsamp),
                    np.linspace(p0[1], p1[1], numsamp),
                    np.linspace(p0[2], p1[2], numsamp),
                ]
            ).T
            sample2 = np.vstack(
                [
                    np.linspace(p1[0], p2[0], numsamp),
                    np.linspace(p1[1], p2[1], numsamp),
                    np.linspace(p1[2], p2[2], numsamp),
                ]
            ).T
            test_sub.append(np.round(sample))
            test_sub.append(np.round(sample2))

        if test_sub:
            test_sub = np.vstack(test_sub).astype(int)
            test_sub = np.maximum(test_sub, 1)
            test_sub = np.minimum(test_sub, np.array(fdims, dtype=int))
            test_ind = np.ravel_multi_index(
                (test_sub[:, 0] - 1, test_sub[:, 1] - 1, test_sub[:, 2] - 1), fdims, order="F"
            )

            cell_bump = cell_borders2.copy()
            _set_lin(cell_bump, cell_body, True)
            _set_lin(cell_bump, test_ind + 1, True)

            bump_idx = np.column_stack(np.nonzero(cell_bump))
            cell_min2 = bump_idx.min(axis=0) + 1
            cell_max2 = bump_idx.max(axis=0) + 1

            numdiff = int(cell_bump.sum())
            while numdiff > 0:
                cell_crop2 = cell_bump[
                    cell_min2[0] - 1 : cell_max2[0],
                    cell_min2[1] - 1 : cell_max2[1],
                    cell_min2[2] - 1 : cell_max2[2],
                ]
                cell_diff2 = (
                    cell_crop2[:-2, 1:-1, 1:-1]
                    + cell_crop2[2:, 1:-1, 1:-1]
                    + cell_crop2[1:-1, :-2, 1:-1]
                    + cell_crop2[1:-1, 2:, 1:-1]
                    + cell_crop2[1:-1, 1:-1, :-2]
                    + cell_crop2[1:-1, 1:-1, 2:]
                )
                inner2 = cell_crop2[1:-1, 1:-1, 1:-1]
                cell_crop2[1:-1, 1:-1, 1:-1] = (cell_diff2 >= 4) | inner2
                numdiff = int(cell_crop2.sum() - cell_bump.sum())
                cell_bump[
                    cell_min2[0] - 1 : cell_max2[0],
                    cell_min2[1] - 1 : cell_max2[1],
                    cell_min2[2] - 1 : cell_max2[2],
                ] = cell_crop2

            cell_processed |= cell_bump

    output = np.flatnonzero(cell_processed.ravel(order="F")) + 1
    return output


def nodes_to_conn(nodes: list) -> list:
    """Translate nodesToConn.m to Python."""

    ends = [i for i, n in enumerate(nodes) if len(n.get("conn", [])) == 1]
    n_nodes = len(nodes)
    connmat = np.zeros((n_nodes, n_nodes), dtype=float)

    for end_idx in ends:
        curr_node = end_idx
        while nodes[curr_node].get("root", 0) not in (0, [], None):
            root = nodes[curr_node]["root"]
            prev = connmat[nodes[curr_node]["num"], root]
            connmat[nodes[curr_node]["num"], root] = np.sqrt(prev**2 + nodes[end_idx].get("misc", 0) ** 2)
            curr_node = root

    x_idx, y_idx = np.nonzero(connmat)
    conn = []
    for x, y in zip(x_idx, y_idx):
        conn.append(generate_connection(int(x), int(y), connmat[x, y], [], []))
    return conn


def grow_capillaries(nodes: list, conn: list, neur_ves: np.ndarray, nv: Dict[str, Any], vp: Dict[str, Any], vres: float) -> Tuple[list, list, Dict[str, Any]]:
    """Translate growCapillaries.m with simplified pure-NumPy logic."""

    rng = np.random.default_rng()
    dilrad = int(np.ceil(vp["mindists"][0] / vres))
    grid = np.arange(-dilrad, dilrad + 1)
    x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")
    se = np.exp(-2 * (x**2 + y**2 + z**2) / max(dilrad**2, 1))
    tmpvol = np.zeros(nv["szum"], dtype=np.float32)

    for edge in conn:
        locs = np.ceil(np.array(edge.get("locs", [])) / vres).astype(int)
        if locs.size == 0:
            continue
        locs = locs[:: max(1, int(np.floor(dilrad / 3))), :]
        for loc in locs:
            tmps = loc - dilrad
            tmps = (tmps < 1) * (1 - tmps)
            tmpu = loc + dilrad
            tmpu = (tmpu > nv["szum"]) * (tmpu - nv["szum"])
            sub = se[tmps[0] : se.shape[0] - tmpu[0], tmps[1] : se.shape[1] - tmpu[1], tmps[2] : se.shape[2] - tmpu[2]]
            low = loc - dilrad + tmps
            high = loc + dilrad - tmpu
            tmpvol[low[0] - 1 : high[0], low[1] - 1 : high[1], low[2] - 1 : high[2]] = np.maximum(
                tmpvol[low[0] - 1 : high[0], low[1] - 1 : high[1], low[2] - 1 : high[2]],
                sub,
            )

    capppos, _ = pseudo_rand_sample3d(nv["size"] / vres, nv["ncapp"], vp["mindists"][2] / vres, vp["sepweight"], (1 - tmpvol).astype(np.float32))
    capppos = capppos * vres + 1 - rng.integers(1, int(vres) + 1, size=capppos.shape)

    nv_vert_conn = rng.integers(1, int(np.ceil(nv["szum"][2] / vp["vesFreq"][2])) + 1, size=nv["nvert"])
    nv["nvert_sum"] = int(np.sum(nv_vert_conn))
    node_idx = nv["nnodes"]
    conn_idx = nv["nconn"]
    vertidxs = [i for i, n in enumerate(nodes) if n.get("type") == "sfvt"]
    if len(vp["vesSize"]) < 4:
        vp["vesSize"] = list(vp["vesSize"]) + [vp["vesSize"][2] * 0]

    for i, ves_idx in enumerate(vertidxs):
        branch = [ves_idx]
        flag = True
        while flag:
            tmpidx = nodes[branch[-1]].get("conn", [])
            tmpidx = [t for t in tmpidx if nodes[t].get("type") == "vert"]
            tmpidx = [t for t in tmpidx if t not in branch]
            if not tmpidx:
                flag = False
            else:
                branch.append(tmpidx[0])
        branch = branch[1:]
        for _ in range(nv_vert_conn[i]):
            if not branch:
                break
            tmpidx = rng.choice(branch)
            node_idx += 1
            conn_idx += 1
            if capppos.size:
                dists = np.sum((capppos - nodes[tmpidx]["pos"]) ** 2, axis=1)
                tmp = int(np.argmin(dists))
            else:
                continue
            nodes.append(generate_node(node_idx, tmpidx, tmpidx, capppos[tmp], "capp", {}))
            nodes[tmpidx]["conn"] = list(set(nodes[tmpidx]["conn"]) | {node_idx})
            weight = max(1.0, rng.normal(vp["vesSize"][2], vp["vesSize"][3]))
            conn.append(generate_connection(node_idx, tmpidx, weight, [], "vtcp"))
            capppos[tmp] = np.nan

    nv["nnodes"] = node_idx
    nv["nconn"] = conn_idx

    node_idx = nv["nnodes"]
    for pos in capppos:
        if not np.any(np.isnan(pos)):
            node_idx += 1
            nodes.append(generate_node(node_idx, [], [], pos, "capp", {}))
    nv["nnodes"] = node_idx

    vertconnidxs = [i for i, n in enumerate(nodes) if n.get("type") == "capp" and n.get("root")]
    cappconnidxs = [i for i, n in enumerate(nodes) if n.get("type") == "capp" and not n.get("root")]
    connidxs = vertconnidxs + cappconnidxs
    capppos = np.array([nodes[i]["pos"] for i in connidxs], dtype=float)
    if capppos.size == 0:
        return nodes, conn, nv

    cappmat = _pos_to_dists(capppos)
    np.fill_diagonal(cappmat, np.inf)
    cappmat[: nv["nvert_sum"], : nv["nvert_sum"]] = np.inf

    cappconnmat = np.zeros((nv["ncapp"], nv["ncapp"]), dtype=int)
    mincapp = np.argmin(cappmat, axis=1)
    cappconnmat[np.arange(nv["ncapp"]), mincapp] = 1
    cappconnmat[mincapp, np.arange(nv["ncapp"])] = 1
    cappmat[np.arange(nv["ncapp"]), mincapp] = np.inf
    cappmat[mincapp, np.arange(nv["ncapp"])] = np.inf

    cappmat[cappmat > vp["maxcappdist"]] = np.inf
    for i in range(nv["ncapp"]):
        if cappconnmat[i].sum() >= 3:
            cappmat[i, :] = np.inf
            cappmat[:, i] = np.inf
        cappmat[i, cappconnmat[i].astype(bool)] = np.inf
        cappmat[cappconnmat[i].astype(bool), i] = np.inf

    for i in range(nv["nvert_sum"], nv["ncapp"]):
        for j in range(i + 1, nv["ncapp"]):
            if np.isfinite(cappmat[i, j]):
                steps = int(max(2, 2 * cappmat[i, j]))
                xpix = np.ceil(np.column_stack([
                    np.linspace(capppos[i, 0], capppos[j, 0], steps),
                    np.linspace(capppos[i, 1], capppos[j, 1], steps),
                    np.linspace(capppos[i, 2], capppos[j, 2], steps),
                ])).astype(int)
                xpix = np.clip(xpix, 1, np.array(nv["size"]))
                if np.any(neur_ves[xpix[:, 0] - 1, xpix[:, 1] - 1, xpix[:, 2] - 1]):
                    cappmat[i, j] = np.inf
                    cappmat[j, i] = np.inf

    lflag = True
    while lflag:
        cappsum = cappconnmat.sum(axis=0)
        if np.min(cappsum[nv["nvert_sum"] :]) > 1:
            lflag = False
            break
        idxs = np.where(cappsum == 1)[0]
        if idxs.size == 0 or np.isinf(cappmat[:, idxs]).all():
            lflag = False
            break
        rndidx = rng.choice(idxs)
        if np.min(cappmat[rndidx]) < np.inf:
            capdistinv = 1.0 / (cappmat[rndidx] ** vp["distsc"])
            capdistinv[~np.isfinite(capdistinv)] = 0
            capcdf = np.cumsum(capdistinv)
            if capcdf[-1] == 0:
                lflag = False
                break
            capcdf = capcdf / capcdf[-1]
            lnkidx = int(np.searchsorted(capcdf, rng.random()))
            cappconnmat[rndidx, lnkidx] = 1
            cappconnmat[lnkidx, rndidx] = 1
            cappmat[cappconnmat[rndidx].astype(bool), lnkidx] = np.inf
            cappmat[cappconnmat[lnkidx].astype(bool), rndidx] = np.inf
            cappmat[lnkidx, cappconnmat[rndidx].astype(bool)] = np.inf
            cappmat[rndidx, cappconnmat[lnkidx].astype(bool)] = np.inf
            cappmat[rndidx, lnkidx] = np.inf
            cappmat[lnkidx, rndidx] = np.inf
            if cappconnmat[lnkidx].sum() >= 3:
                cappmat[lnkidx] = np.inf
                cappmat[:, lnkidx] = np.inf

    conn_s, conn_f = np.where(np.triu(cappconnmat))
    conn_idx = nv["nconn"]
    conn_mat = np.zeros((len(nodes), len(nodes)), dtype=int)
    for s, f in zip(conn_s, conn_f):
        nodes[connidxs[s]]["conn"] = list(set(nodes[connidxs[s]]["conn"]) | {connidxs[f]})
        nodes[connidxs[f]]["conn"] = list(set(nodes[connidxs[f]]["conn"]) | {connidxs[s]})
        conn_idx += 1
        conn.append(generate_connection(connidxs[s], connidxs[f], np.nan, [], "capp"))
        conn_mat[connidxs[s], connidxs[f]] = conn_idx

    nv["nconn"] = conn_idx
    nodes_to_connect = [c["start"] for c in conn if c.get("misc") == "vtcp"]
    to_connect: list[int] = []
    for n in nodes_to_connect:
        for j in nodes[n]["conn"]:
            if conn_mat[n, j]:
                to_connect.append(conn_mat[n, j])
    to_connect = list(np.unique(to_connect))
    for i, c in enumerate(conn):
        if c.get("misc") == "vtcp":
            conn_mat[c["ends"], c["start"]] = i + 1
    conn_mat = conn_mat + conn_mat.T

    while to_connect:
        curr_conn = to_connect.pop(0)
        curr_idx = curr_conn - 1
        if np.isnan(conn[curr_idx]["weight"]):
            conn_start = conn[curr_idx]["start"]
            conn_end = conn[curr_idx]["ends"]
            start_conns = [c for c in conn_mat[conn_start] if c]
            end_conns = [c for c in conn_mat[conn_end] if c]
            start_conns = [c - 1 for c in start_conns if c - 1 != curr_idx]
            end_conns = [c - 1 for c in end_conns if c - 1 != curr_idx]
            start_weights = [conn[c]["weight"] for c in start_conns]
            end_weights = [conn[c]["weight"] for c in end_conns]

            weight1 = np.nan
            weight2 = np.nan
            if start_weights and not np.isnan(start_weights).any():
                if len(start_weights) == 1:
                    weight1 = start_weights[0]
                else:
                    t1 = max(start_weights) ** 2 - min(start_weights) ** 2
                    t2 = max(start_weights) ** 2 + min(start_weights) ** 2
                    weight1 = np.sqrt(rng.random() * (t2 - t1) + t1)
            if end_weights and not np.isnan(end_weights).any():
                if len(end_weights) == 1:
                    weight2 = end_weights[0]
                else:
                    t1 = max(end_weights) ** 2 - min(end_weights) ** 2
                    t2 = max(end_weights) ** 2 + min(end_weights) ** 2
                    weight2 = np.sqrt(rng.random() * (t2 - t1) + t1)

            if np.isnan(weight1) and np.isnan(weight2):
                conn_weight = max(1.0, rng.normal(vp["vesSize"][2], vp["vesSize"][3]))
            elif np.isnan(weight1):
                conn_weight = weight2
            elif np.isnan(weight2):
                conn_weight = weight1
            else:
                conn_weight = (weight1 + weight2) / 2
            conn[curr_idx]["weight"] = conn_weight
            for c in end_conns:
                if np.isnan(conn[c]["weight"]):
                    to_connect.append(c + 1)
            for c in start_conns:
                if np.isnan(conn[c]["weight"]):
                    to_connect.append(c + 1)

    for i in range(nv["nconn"]):
        if conn[i].get("weight") is None or np.isnan(conn[i]["weight"]):
            conn[i]["weight"] = max(1.0, rng.normal(vp["vesSize"][2], vp["vesSize"][3]))

    return nodes, conn, nv


def grow_major_vessels(nv: Dict[str, Any], np_nodes: Dict[str, Any], vp: Dict[str, Any]) -> Tuple[list, Dict[str, Any]]:
    """Translate growMajorVessels.m with simplified pure-NumPy logic."""

    rng = np.random.default_rng()
    nodes: list = []

    for i in range(nv["nsource"]):
        tmpidx = rng.random(2) >= np.array([(nv["size"][1] / (nv["size"][0] + nv["size"][1])), 0.5])
        if tmpidx[0] and tmpidx[1]:
            tmppos = [rng.integers(1, nv["size"][0] + 1), 1, vp["depth_surf"]]
        elif tmpidx[0] and not tmpidx[1]:
            tmppos = [rng.integers(1, nv["size"][0] + 1), nv["size"][1], vp["depth_surf"]]
        elif not tmpidx[0] and tmpidx[1]:
            tmppos = [1, rng.integers(1, nv["size"][1] + 1), vp["depth_surf"]]
        else:
            tmppos = [nv["size"][0], rng.integers(1, nv["size"][1] + 1), vp["depth_surf"]]
        nodes.append(generate_node(i, 0, [], tmppos, "edge", tmpidx.tolist()))

    neur_surf = np.zeros(nv["size"][:2], dtype=bool)
    for i in range(nv["nsource"]):
        misc = nodes[i]["misc"]
        if misc[0] and misc[1]:
            rand_dir = 0.5 * np.pi + rng.standard_normal() * np_nodes["dirvar"]
        elif misc[0] and not misc[1]:
            rand_dir = 1.5 * np.pi + rng.standard_normal() * np_nodes["dirvar"]
        elif not misc[0] and misc[1]:
            rand_dir = 0.0 * np.pi + rng.standard_normal() * np_nodes["dirvar"]
        else:
            rand_dir = 1.0 * np.pi + rng.standard_normal() * np_nodes["dirvar"]
        nodes, neur_surf = branch_grow_nodes(nodes, neur_surf, np_nodes, i, rand_dir)

    nv["nlinks"] = len(nodes) - nv["nsource"]
    for i in range(nv["nsource"], nv["nlinks"] + nv["nsource"]):
        pos = nodes[i]["pos"]
        nodes[i]["pos"] = [int(round(pos[0])), int(round(pos[1])), vp["depth_surf"]]

    dilrad = int(round(vp["mindists"][0] * 2))
    neur_surf = _binary_dilate(neur_surf, _disk_structuring_element(dilrad))

    surfpos, _ = pseudo_rand_sample2d(
        tuple(nv["size"][:2]),
        nv["nsurf"],
        vp["mindists"][0],
        vp["sepweight"],
        (1 - neur_surf).astype(np.float32),
        100,
    )
    surfpos = np.column_stack([surfpos, np.full(nv["nsurf"], vp["depth_surf"])])
    seed_positions = np.array([n["pos"] for n in nodes[: nv["nlinks"] + nv["nsource"]]])
    surfpos = np.vstack([seed_positions, surfpos])

    surfmat = _pos_to_dists(surfpos.astype(float))
    surfmat[: nv["nlinks"] + nv["nsource"], : nv["nlinks"] + nv["nsource"]] = np.inf
    surfmat[: nv["nsource"], : nv["nsource"]] = 0
    for i in range(nv["nlinks"] + nv["nsource"]):
        if nodes[i]["root"] > 0:
            root_idx = nodes[i]["root"]
            surfmat[root_idx, i] = np.linalg.norm(np.array(nodes[i]["pos"]) - np.array(nodes[root_idx]["pos"]))
            surfmat[i, root_idx] = surfmat[root_idx, i]
    surfmat[nv["nlinks"] + nv["nsource"] :, : nv["nlinks"] + nv["nsource"]] = np.inf

    tmp_surfmat = (surfmat ** vp["distWeightScale"]) * (1 + vp["randWeightScale"] * rng.standard_normal(surfmat.shape))
    _, surfpath = vessel_dijkstra(tmp_surfmat, 0)
    surfpath[: nv["nsource"]] = np.arange(nv["nsource"])
    for i in range(nv["nsurf"] + nv["nsource"]):
        if surfpath[i] == 0:
            surfpath[i] = int(np.argmin(surfmat[i, : nv["nsource"]]))

    for i in range(nv["nlinks"] + nv["nsource"], nv["nlinks"] + nv["nsource"] + nv["nsurf"]):
        pos = surfpos[i]
        if pos[0] in (1, nv["size"][0]) or pos[1] in (1, nv["size"][1]):
            nodes.append(generate_node(i, int(surfpath[i]), int(surfpath[i]), pos, "edge", []))
        else:
            nodes.append(generate_node(i, int(surfpath[i]), int(surfpath[i]), pos, "surf", []))

    nv["nnodes"] = nv["nlinks"] + nv["nsource"] + nv["nsurf"]
    for i in range(nv["nnodes"]):
        if nodes[i]["root"] != 0:
            root_idx = int(nodes[i]["root"])
            nodes[root_idx]["conn"] = list(set(nodes[root_idx]["conn"]) | {i})

    neur_vert = np.zeros(nv["size"][:2], dtype=bool)
    se = _disk_structuring_element(int(round(vp["mindists"][0] * 2)))
    for i in range(nv["nnodes"]):
        if nodes[i]["type"] == "surf" and len(nodes[i]["conn"]) == 1:
            pos = nodes[i]["pos"]
            if not neur_vert[int(pos[0]) - 1, int(pos[1]) - 1]:
                nodes[i]["type"] = "sfvt"
                tmp = np.zeros(nv["size"][:2], dtype=bool)
                tmp[int(pos[0]) - 1, int(pos[1]) - 1] = True
                neur_vert |= _binary_dilate(tmp, se)
            else:
                nodes[i] = generate_node(i, [], [], [0, 0, 0], "", {})

    nodes = nodes[: nv["nnodes"]]
    surfidx = [i for i, n in enumerate(nodes) if n["type"] == "surf"]
    if surfidx:
        surfpos = np.array([nodes[i]["pos"] for i in surfidx])
        surfmask = (neur_vert[surfpos[:, 0].astype(int) - 1, surfpos[:, 1].astype(int) - 1] == 0)
        while sum(n["type"] == "sfvt" for n in nodes) < nv["nvert"] and np.any(surfmask):
            candidates = np.array(surfidx)[np.where(surfmask)[0]]
            tmpidx = int(rng.choice(candidates))
            nodes[tmpidx]["type"] = "sfvt"
            tmp = np.zeros(nv["size"][:2], dtype=bool)
            pos = nodes[tmpidx]["pos"]
            tmp[int(pos[0]) - 1, int(pos[1]) - 1] = True
            neur_vert |= _binary_dilate(tmp, se)
            surfmask = neur_vert[surfpos[:, 0].astype(int) - 1, surfpos[:, 1].astype(int) - 1] == 0

    vertidx = [i for i, n in enumerate(nodes) if n["type"] == "sfvt"]
    tmpidx = nv["nnodes"]
    for vidx in vertidx:
        curr_node = vidx
        while nodes[curr_node]["pos"][2] < nv["size"][2]:
            tmpidx += 1
            node_pos = np.array(nodes[curr_node]["pos"]) + np.ceil(
                np.array([rng.standard_normal(), rng.standard_normal(), 1])
                * np.array([np_nodes["varpos"], np_nodes["varpos"], max(np_nodes["varsc"] * rng.standard_normal() + np_nodes["lensc"], np_nodes["mindist"])])
            )
            node_pos = np.clip(node_pos, [1, 1, 1], nv["size"])
            nodes.append(generate_node(tmpidx, curr_node, curr_node, node_pos, "vert", []))
            nodes[curr_node]["conn"] = list(set(nodes[curr_node]["conn"]) | {tmpidx})
            curr_node = tmpidx
    nv["nvert"] = sum(1 for n in nodes if n["type"] == "sfvt")
    nv["nvertconn"] = curr_node - nv["nnodes"] if vertidx else 0
    nv["nnodes"] = curr_node if vertidx else nv["nnodes"]

    ends = [i for i, n in enumerate(nodes) if len(n.get("conn", [])) == 1]
    for idx in ends:
        nodes[idx]["misc"] = vp["vesSize"][2] + rng.gamma(3, (vp["vesSize"][1] - vp["vesSize"][2]) / 3)

    return nodes, nv


def vessel_dijkstra(distMat: np.ndarray, proot: int) -> Tuple[np.ndarray, np.ndarray]:
    """Translate vessel_dijkstra.m to Python."""

    dist_mat = np.asarray(distMat, dtype=float)
    dims = dist_mat.shape[0]
    to_visit = np.ones(dims, dtype=bool)
    unvisited = np.zeros(dims, dtype=float)
    root = int(proot)
    unvisited[root] = 1.0
    distance = np.full(dims, np.inf, dtype=float)
    distance[root] = 0.0
    pathfrom = np.full(dims, np.nan, dtype=float)
    cn = root

    while np.count_nonzero(unvisited):
        to_visit[cn] = False
        nextidx = np.flatnonzero(unvisited)
        idx = int(np.argmin(unvisited[nextidx]))
        cn = int(nextidx[idx])
        unvisited[cn] = 0.0
        for nn in range(dims):
            if to_visit[nn]:
                ndist = distance[cn] + dist_mat[cn, nn]
                if ndist < distance[nn]:
                    unvisited[nn] = max(np.finfo(float).eps, ndist)
                    distance[nn] = ndist
                    pathfrom[nn] = cn

    return distance, pathfrom
