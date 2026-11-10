import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("pycolmap")
from insv_odometry.recon import fuse  # noqa: E402


def test_kabsch_recovers_rotation():
    rng = np.random.default_rng(1)
    R = Rotation.from_euler("xyz", [10, -80, 35], degrees=True).as_matrix()
    src = rng.normal(size=(50, 3))
    assert np.allclose(fuse.kabsch(src, src @ R.T), R, atol=1e-9)


def test_triangle_mean_equals_second_difference():
    # p(t) = sin(3t): second difference over h == triangle-weighted mean of p''(t)
    t = np.arange(0, 5, 0.001)
    acc = (-9 * np.sin(3 * t))[:, None]
    h, tc = 0.2, np.array([1.0, 2.5, 3.7])
    p = lambda x: np.sin(3 * x)
    sd = (p(tc + h) - 2 * p(tc) + p(tc - h)) / h**2
    tm = fuse.triangle_mean(t, acc, tc, h)[:, 0]
    assert np.allclose(sd, tm, atol=2e-3)


def test_time_offset_recovered():
    # IMU sees a rotation that the "camera" sees 40 ms later
    t_imu = np.arange(0, 10, 0.001)
    w = np.stack([np.sin(2 * t_imu), np.cos(1.3 * t_imu), 0.5 * np.sin(0.7 * t_imu + 1)], 1)
    from insv_odometry import orientation

    acc = np.tile([0, 0, 9.80665], (len(t_imu), 1))
    q = orientation.mahony(t_imu, w, acc, kp=0, ki=0)
    t_cam = np.arange(1, 9, 0.2)
    true_dt = 0.040
    R_sc = orientation.to_scipy(orientation.resample(t_imu, q, t_cam + true_dt)).as_matrix()
    dt, _, _ = fuse.estimate_time_offset(t_imu, q, t_cam, R_sc)
    assert abs(dt - true_dt) < 0.002


def test_consistent_segments_splits_at_jump():
    resid = np.array([0.3, 0.5, 40.0, 0.4, 0.2, 0.6, 0.3])  # 8 frames, jump between frames 2 and 3
    segs = fuse.consistent_segments(resid)
    assert segs[0] == (3, 8) and segs[1] == (0, 3)


def test_robust_hand_eye_ignores_outlier_steps():
    rng = np.random.default_rng(3)
    R = Rotation.from_euler("xyz", [5, 88, -3], degrees=True).as_matrix()
    r_c = rng.normal(scale=0.1, size=(80, 3))
    r_i = r_c @ R.T
    r_i[[10, 40]] += 0.7  # two gross outliers (~40 deg)
    R_est, resid, inl = fuse.robust_hand_eye(r_c, r_i)
    assert np.degrees(Rotation.from_matrix(R_est.T @ R).magnitude()) < 0.01
    assert not inl[10] and not inl[40]


def test_longest_drift_free_excludes_drifting_part():
    n = 60
    R_i = Rotation.from_euler("z", np.linspace(0, 90, n)[:, None], degrees=True).as_matrix()
    drift = np.zeros(n)
    drift[20:30] = np.linspace(3, 30, 10)  # 10 steps of ~3 deg extra SfM rotation
    drift[30:] = 30
    R_sc = Rotation.from_euler("x", drift[:, None], degrees=True).as_matrix() @ R_i  # world-side drift
    a, b = fuse.longest_drift_free(R_sc, R_i, np.eye(3), 0, n, 15.0)
    assert b == n and 25 <= a <= 30  # kept the long stable tail, not the drifting stretch
