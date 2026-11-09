"""Write extraction results to disk."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .orientation import to_scipy
from .parse import Telemetry
from .video import VideoTrack


def _write_csv(path: Path, header: list[str], rows: np.ndarray, fmt: str = "%.9g") -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([fmt % v if isinstance(v, (float, np.floating)) else v for v in r])


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def euler_deg(q_wxyz: np.ndarray) -> np.ndarray:
    """Intrinsic Z-Y-X (yaw, pitch, roll) in degrees, returned as [roll, pitch, yaw]."""
    ypr = to_scipy(q_wxyz).as_euler("ZYX", degrees=True)
    return ypr[:, ::-1]


def rotation_diff_deg(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """Geodesic angle between two orientation series, degrees."""
    return np.rad2deg((to_scipy(qa).inv() * to_scipy(qb)).magnitude())


def write_imu(out: Path, tel: Telemetry) -> None:
    rows = np.column_stack([tel.t, tel.gyro, tel.accel])
    _write_csv(out / "imu.csv",
               ["t_s", "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s",
                "accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2"], rows)


def write_orientation(out: Path, t: np.ndarray, q: np.ndarray, name: str,
                      extra: dict[str, np.ndarray] | None = None) -> None:
    eul = euler_deg(q)
    cols = [t, q, eul]
    header = ["t_s", "qw", "qx", "qy", "qz", "roll_deg", "pitch_deg", "yaw_deg"]
    for k, v in (extra or {}).items():
        cols.append(v)
        header.append(k)
    _write_csv(out / name, header, np.column_stack(cols))


def write_frames(out: Path, track: VideoTrack, q: np.ndarray, exposure_s: np.ndarray) -> None:
    eul = euler_deg(q)
    idx = np.arange(len(track.frame_times))
    rows = np.column_stack([idx, track.frame_times, q, eul, exposure_s])
    header = ["frame", "t_s", "qw", "qx", "qy", "qz",
              "roll_deg", "pitch_deg", "yaw_deg", "exposure_s"]
    with (out / "frames.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([int(r[0])] + ["%.9g" % v for v in r[1:]])


def write_tum(out: Path, t: np.ndarray, q: np.ndarray) -> None:
    """TUM format: t tx ty tz qx qy qz qw. Translation is zero (rotation only)."""
    with (out / "trajectory_tum.txt").open("w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw  (IMU orientation only; translation not observable from IMU)\n")
        for ti, (qw, qx, qy, qz) in zip(t, q):
            f.write(f"{ti:.6f} 0 0 0 {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")


def plot(out: Path, tel: Telemetry, q: np.ndarray, q_gyro_only: np.ndarray) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eul = euler_deg(q)
    diff = rotation_diff_deg(q, q_gyro_only)
    fig, ax = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    for i, c in enumerate("xyz"):
        ax[0].plot(tel.t, np.rad2deg(tel.gyro[:, i]), lw=0.6, label=c)
        ax[1].plot(tel.t, tel.accel[:, i], lw=0.6, label=c)
    ax[1].plot(tel.t, np.linalg.norm(tel.accel, axis=1), "k", lw=0.6, label="|a|")
    ax[0].set_ylabel("gyro [deg/s]")
    ax[1].set_ylabel("accel [m/s²]")
    for i, name in enumerate(["roll", "pitch", "yaw"]):
        ax[2].plot(tel.t, eul[:, i], lw=1, label=name)
    ax[3].plot(tel.t, diff, "k", lw=1, label="angle(Mahony, gyro-only)")
    ax[2].set_ylabel("orientation [deg]\n(Mahony, Z-up)")
    ax[3].set_ylabel("gravity correction\n[deg]")
    ax[3].set_xlabel("time since first video frame [s]")
    for a in ax:
        a.legend(loc="upper right", fontsize=8)
        a.grid(alpha=0.3)
    fig.suptitle(f"{tel.camera} {tel.model}: IMU odometry")
    fig.tight_layout()
    path = out / "odometry.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
