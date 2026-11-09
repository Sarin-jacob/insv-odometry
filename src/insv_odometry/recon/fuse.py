"""Fuse the SfM trajectory with the IMU.

Estimates, in order:
1. IMU<->video time offset `dt`: match per-step rotation angles (SfM vs gyro-only integration).
2. Camera->IMU rotation `R_ic` (hand-eye, rotation only): Kabsch on per-step rotation vectors.
3. Gravity alignment `R_gs` (SfM world -> Z-up world) from the gravity-corrected IMU orientation.
4. Metric scale `s` and accelerometer bias `b`. A second difference of SfM positions over
   step h equals the triangle-weighted mean of true acceleration over [t-h, t+h]. Applying
   the same kernel to the IMU acceleration gives a linear least-squares problem with no
   smoothing mismatch.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pycolmap
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .. import orientation
from ..parse import G, Telemetry
from .lens import FisheyeLens
from .views import VirtualView

_FRAME_RE = re.compile(r"frame_(\d+)")


def sfm_poses(rec: pycolmap.Reconstruction, ref_prefix: str, frame_times: dict[int, float]):
    """Return (t, frame_idx, R_sc (cam->sfm world), c_s (centre)) sorted by time."""
    out = []
    for img in rec.images.values():
        if not img.name.startswith(ref_prefix) or not img.has_pose:
            continue
        idx = int(_FRAME_RE.search(img.name).group(1))
        cw = img.cam_from_world()
        R_cw = cw.rotation.matrix()
        out.append((frame_times[idx], idx, R_cw.T, img.projection_center()))
    out.sort(key=lambda r: r[0])
    t = np.array([r[0] for r in out])
    idx = np.array([r[1] for r in out])
    R_sc = np.stack([r[2] for r in out])
    c = np.stack([r[3] for r in out])
    return t, idx, R_sc, c


def _rel_rotvecs(R: np.ndarray) -> np.ndarray:
    """Body-frame increments R_k^T R_{k+1} as rotation vectors, (N-1, 3)."""
    return Rotation.from_matrix(np.einsum("nji,njk->nik", R[:-1], R[1:])).as_rotvec()


def _imu_R(t_imu, q_wxyz, t):
    return orientation.to_scipy(orientation.resample(t_imu, q_wxyz, t)).as_matrix()


def estimate_time_offset(t_imu, q_gyro, t, R_sc, search=0.3, step=0.001):
    ang_s = np.linalg.norm(_rel_rotvecs(R_sc), axis=1)
    dts = np.arange(-search, search + step / 2, step)
    cost = []
    for dt in dts:
        ang_i = np.linalg.norm(_rel_rotvecs(_imu_R(t_imu, q_gyro, t + dt)), axis=1)
        cost.append(np.mean((ang_s - ang_i) ** 2))
    cost = np.array(cost)
    k = int(np.argmin(cost))
    dt = dts[k]
    if 0 < k < len(dts) - 1:  # parabolic refinement
        a, b, c = cost[k - 1 : k + 2]
        den = a - 2 * b + c
        if den > 0:
            dt += 0.5 * (a - c) / den * step
    return float(dt), dts, cost


def kabsch(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """R minimising sum |dst - R src|^2."""
    H = src.T @ dst
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def triangle_mean(t_imu, values, centres, h):
    """Triangle-kernel (half width h) weighted mean of `values` around each centre."""
    out = np.empty((len(centres),) + values.shape[1:])
    for k, tc in enumerate(centres):
        lo, hi = np.searchsorted(t_imu, [tc - h, tc + h])
        w = np.clip(1 - np.abs(t_imu[lo:hi] - tc) / h, 0, None)
        out[k] = np.tensordot(w, values[lo:hi], axes=(0, 0)) / w.sum()
    return out


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    with path.open("wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        rec = np.zeros(len(xyz), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
        rec["p"], rec["c"] = xyz, rgb
        f.write(rec.tobytes())


def run(work: Path, rec: pycolmap.Reconstruction, tel: Telemetry,
        views: list[VirtualView], lenses: list[FisheyeLens]) -> dict:
    import csv

    with (work / "frames.csv").open() as f:
        frame_times = {int(r["frame"]): float(r["t_s"]) for r in csv.DictReader(f)}
    t, fidx, R_sc, c_s = sfm_poses(rec, f"{views[0].name}/", frame_times)
    n = len(t)
    print(f"Fusion: {n} registered rig frames of {len(frame_times)}")
    if n < 10:
        raise RuntimeError("too few registered frames for IMU fusion")

    q_gyro = orientation.mahony(tel.t, tel.gyro, tel.accel, kp=0.0, ki=0.0)
    q_imu = orientation.mahony(tel.t, tel.gyro, tel.accel)

    # 1. time offset
    dt, dts, cost = estimate_time_offset(tel.t, q_gyro, t, R_sc)
    ti = t + dt

    # 2. hand-eye rotation (camera -> IMU)
    r_c = _rel_rotvecs(R_sc)
    r_i = _rel_rotvecs(_imu_R(tel.t, q_gyro, ti))
    R_ic = kabsch(r_c, r_i)
    resid = np.degrees(np.linalg.norm(r_i - r_c @ R_ic.T, axis=1))
    step_ang = np.degrees(np.linalg.norm(r_i, axis=1))

    # 3. gravity alignment: R_gc = R_gi R_ic, R_gs = R_gc R_sc^T
    R_gi = _imu_R(tel.t, q_imu, ti)
    R_gc = R_gi @ R_ic
    R_gs_all = Rotation.from_matrix(R_gc @ np.transpose(R_sc, (0, 2, 1)))
    R_gs_rot = R_gs_all.mean()
    R_gs = R_gs_rot.as_matrix()
    align_spread = np.degrees((R_gs_all * R_gs_rot.inv()).magnitude())
    p_g = c_s @ R_gs.T  # unscaled positions in gravity world

    # 4. scale + accel bias, on uniformly spaced consecutive triples only
    h = np.median(np.diff(t))
    ok = np.where((np.abs(np.diff(t)[:-1] - h) < 1e-3) & (np.abs(np.diff(t)[1:] - h) < 1e-3))[0] + 1
    A_s = (p_g[ok + 1] - 2 * p_g[ok] + p_g[ok - 1]) / h**2
    R_w = orientation.to_scipy(q_imu).as_matrix()  # IMU->world at IMU rate
    a_world = np.einsum("nij,nj->ni", R_w, tel.accel) - np.array([0, 0, G])
    A_i = triangle_mean(tel.t, a_world, ti[ok], h)
    B = triangle_mean(tel.t, R_w, ti[ok], h)  # world-from-IMU, averaged

    def residual(x):
        s, b = x[0], x[1:]
        return (s * A_s + np.einsum("nij,j->ni", B, b) - A_i).ravel()

    sol = least_squares(residual, x0=[1.0, 0, 0, 0], loss="huber", f_scale=0.5)
    s, bias = float(sol.x[0]), sol.x[1:]
    acc_res = residual(sol.x).reshape(-1, 3)
    acc_rms = float(np.sqrt(np.mean(acc_res**2)))
    acc_sig = float(np.sqrt(np.mean(A_i**2)))
    # scale uncertainty from the Jacobian
    J = sol.jac
    try:
        cov = np.linalg.inv(J.T @ J) * np.mean(sol.fun**2)
        s_std = float(np.sqrt(cov[0, 0]))
    except np.linalg.LinAlgError:
        s_std = float("nan")

    p_m = s * p_g
    path_len = float(np.sum(np.linalg.norm(np.diff(p_m, axis=0), axis=1)))

    # outputs
    q_gc = Rotation.from_matrix(R_gc).as_quat()  # x y z w (camera->world, IMU-gravity based)
    q_sfm = Rotation.from_matrix(R_gs @ R_sc).as_quat()  # camera->world from SfM
    with (work / "trajectory_metric_tum.txt").open("w") as f:
        f.write("# t_video_s x y z qx qy qz qw   (lens0 centre camera, Z-up gravity world, metres)\n")
        for k in range(n):
            f.write(f"{t[k]:.6f} {p_m[k, 0]:.5f} {p_m[k, 1]:.5f} {p_m[k, 2]:.5f} "
                    + " ".join(f"{v:.8f}" for v in q_sfm[k]) + "\n")

    pts = np.array([p.xyz for p in rec.points3D.values()])
    cols = np.array([p.color for p in rec.points3D.values()], dtype=np.uint8)
    errs = np.array([p.error for p in rec.points3D.values()])
    keep = errs < 2.0
    write_ply(work / "sparse_points_metric.ply", s * (pts[keep] @ R_gs.T), cols[keep])

    res = {
        "registered_frames": int(n),
        "total_frames": len(frame_times),
        "sfm_points": int(len(pts)),
        "sfm_mean_reproj_px": float(rec.compute_mean_reprojection_error()),
        "time_offset_s": dt,
        "R_cam_to_imu": {
            "quat_xyzw": Rotation.from_matrix(R_ic).as_quat().tolist(),
            "euler_xyz_deg": Rotation.from_matrix(R_ic).as_euler("xyz", degrees=True).tolist(),
        },
        "rotation_residual_deg": {"rms": float(np.sqrt(np.mean(resid**2))),
                                  "median": float(np.median(resid)), "max": float(resid.max()),
                                  "mean_step_rotation_deg": float(step_ang.mean())},
        "gravity_alignment_spread_deg": {"median": float(np.median(align_spread)),
                                         "max": float(align_spread.max())},
        "scale": {"value": s, "std": s_std,
                  "note": "SfM units -> metres; SfM is already ~metric if the fixed 3.35 cm "
                          "lens offset constrained it, then s ~ 1"},
        "accel_bias_m_s2": bias.tolist(),
        "accel_fit": {"rms_residual_m_s2": acc_rms, "rms_signal_m_s2": acc_sig,
                      "samples": int(len(ok))},
        "path_length_m": path_len,
        "extent_m": (p_m.max(0) - p_m.min(0)).tolist(),
    }
    (work / "fusion.json").write_text(json.dumps(res, indent=2))
    _plot(work, t, dts, cost, dt, step_ang, resid, A_s * s, A_i - np.einsum("nij,j->ni", B, bias),
          t[ok], p_m, s * (pts[keep] @ R_gs.T))
    _plot_cloud(work, s * (pts[keep] @ R_gs.T), cols[keep], p_m)
    print(json.dumps(res, indent=2))
    return res


def _plot_cloud(work, pts, cols, p_m, radius=8.0):
    """Top and side views of the coloured metric cloud around the trajectory."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = np.median(p_m, 0)
    near = np.linalg.norm(pts[:, :2] - c[:2], axis=1) < radius
    P, C = pts[near], cols[near] / 255.0
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    for a, (i, j, lab) in zip(ax, [(0, 1, "top (x-y)"), (0, 2, "side (x-z)")]):
        order = np.argsort(P[:, 2] if j != 2 else -P[:, 1])  # draw far points first
        a.scatter(P[order, i], P[order, j], c=C[order], s=1.5, linewidths=0)
        a.plot(p_m[:, i], p_m[:, j], "r-", lw=2, label="camera")
        a.set_aspect("equal")
        a.set_title(f"metric sparse cloud, {lab} [m]")
        a.set_facecolor("0.15")
        a.legend()
    fig.tight_layout()
    fig.savefig(work / "cloud.png", dpi=110)
    plt.close(fig)


def _plot(work, t, dts, cost, dt, step_ang, resid, A_s, A_i, t_ok, p_m, pts):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(2, 3, 1)
    ax.plot(dts * 1000, np.degrees(np.sqrt(cost)))
    ax.axvline(dt * 1000, color="r", ls="--")
    ax.set(xlabel="IMU time offset [ms]", ylabel="rms step-angle diff [deg]", title=f"sync: {dt*1000:.1f} ms")
    ax = fig.add_subplot(2, 3, 2)
    tm = 0.5 * (t[1:] + t[:-1])
    ax.plot(tm, step_ang, label="IMU step rotation")
    ax.plot(tm, resid, label="|SfM − IMU| after hand-eye")
    ax.set(xlabel="t [s]", ylabel="deg per step", title="rotation agreement")
    ax.legend(fontsize=8)
    ax = fig.add_subplot(2, 3, 3)
    for i, cname in enumerate("xyz"):
        ax.plot(t_ok, A_i[:, i], color=f"C{i}", lw=1, label=f"IMU {cname}")
        ax.plot(t_ok, A_s[:, i], color=f"C{i}", lw=1, ls="--", label=f"SfM×s {cname}")
    ax.set(xlabel="t [s]", ylabel="m/s²", title="acceleration (scale fit)")
    ax.legend(fontsize=7, ncol=2)
    ax = fig.add_subplot(2, 3, (4, 5))
    sub = pts[np.random.default_rng(0).choice(len(pts), min(len(pts), 20000), replace=False)]
    ax.scatter(sub[:, 0], sub[:, 1], s=0.3, c="0.6")
    ax.plot(p_m[:, 0], p_m[:, 1], "r.-", ms=3)
    ax.plot(*p_m[0, :2], "go", label="start")
    ax.set(xlabel="x [m]", ylabel="y [m]", title="top view (Z up)")
    ax.set_aspect("equal")
    lim = np.percentile(np.abs(sub[:, :2] - np.median(p_m[:, :2], 0)), 90) * 1.2
    cx, cy = np.median(p_m[:, :2], 0)
    ax.set_xlim(cx - lim, cx + lim)
    ax.set_ylim(cy - lim, cy + lim)
    ax.legend(fontsize=8)
    ax = fig.add_subplot(2, 3, 6)
    ax.plot(t, p_m[:, 2] - p_m[0, 2])
    ax.set(xlabel="t [s]", ylabel="height change [m]", title="camera height")
    fig.tight_layout()
    fig.savefig(work / "fusion.png", dpi=100)
    plt.close(fig)
