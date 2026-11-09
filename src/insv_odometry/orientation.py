"""Orientation estimate from gyro + accelerometer (Mahony complementary filter).

World frame: Z up (gravity along -Z), yaw = 0 at the first sample. Roll and
pitch are corrected by gravity; yaw has no absolute reference (no
magnetometer) and drifts slowly with residual gyro bias.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .parse import G


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, [w, x, y, z]."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def _initial_quat(accel: np.ndarray) -> np.ndarray:
    """Rotation body->world that maps the measured 'up' vector onto +Z (yaw = 0)."""
    up = accel / np.linalg.norm(accel)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(up, z)
    s = np.linalg.norm(axis)
    c = float(np.dot(up, z))
    if s < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0]) if c > 0 else np.array([0.0, 1.0, 0.0, 0.0])
    angle = np.arctan2(s, c)
    axis /= s
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


def estimate_gyro_bias(t: np.ndarray, gyro: np.ndarray, window_s: float = 0.5,
                       still_thresh: float = 0.02) -> np.ndarray:
    """Mean gyro over the stillest window, if it is still enough; else zeros."""
    rate = 1.0 / np.median(np.diff(t))
    n = max(int(window_s * rate), 10)
    if len(gyro) < n:
        return np.zeros(3)
    mag = np.linalg.norm(gyro, axis=1)
    csum = np.concatenate([[0.0], np.cumsum(mag)])
    means = (csum[n:] - csum[:-n]) / n
    i = int(np.argmin(means))
    if means[i] > still_thresh:  # rad/s: never really still
        return np.zeros(3)
    return gyro[i : i + n].mean(axis=0)


def mahony(t: np.ndarray, gyro: np.ndarray, accel: np.ndarray, kp: float = 1.0,
           ki: float = 0.01, accel_gate: float = 0.15,
           bias: np.ndarray | None = None) -> np.ndarray:
    """Return (N, 4) quaternions [w, x, y, z], body(IMU)->world."""
    n = len(t)
    q = np.empty((n, 4))
    gyro = gyro - (bias if bias is not None else 0.0)

    # Initialise roll/pitch from the mean of the first 50 ms of accel.
    n0 = max(1, int(np.searchsorted(t, t[0] + 0.05)))
    q[0] = _initial_quat(accel[:n0].mean(axis=0))
    integral = np.zeros(3)

    for k in range(1, n):
        dt = t[k] - t[k - 1]
        qk = q[k - 1]
        w = gyro[k].copy()
        a = accel[k]
        an = np.linalg.norm(a)
        if abs(an - G) < accel_gate * G:
            a = a / an
            qw, qx, qy, qz = qk
            # world +Z expressed in body frame (third row of R(q))
            v = np.array(
                [
                    2 * (qx * qz - qw * qy),
                    2 * (qw * qx + qy * qz),
                    qw * qw - qx * qx - qy * qy + qz * qz,
                ]
            )
            e = np.cross(a, v)
            integral += ki * e * dt
            w += kp * e + integral
        dq = 0.5 * _quat_mul(qk, np.array([0.0, *w]))
        qn = qk + dq * dt
        q[k] = qn / np.linalg.norm(qn)
    return q


def to_scipy(q_wxyz: np.ndarray) -> Rotation:
    return Rotation.from_quat(q_wxyz[:, [1, 2, 3, 0]])


def resample(t: np.ndarray, q_wxyz: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    """Slerp orientation at new timestamps (clamped to the IMU range)."""
    slerp = Slerp(t, to_scipy(q_wxyz))
    r = slerp(np.clip(t_new, t[0], t[-1]))
    xyzw = r.as_quat()
    return xyzw[:, [3, 0, 1, 2]]
