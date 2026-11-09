import numpy as np
from scipy.spatial.transform import Rotation

from insv_odometry import orientation
from insv_odometry.parse import G


def _synthetic(rate=1000.0, dur=10.0):
    """Body spins at a known rate about a tilted axis; accel = gravity seen in body."""
    t = np.arange(0, dur, 1 / rate)
    w_body = np.array([0.3, -0.2, 0.5])  # rad/s, constant in body frame
    r0 = Rotation.from_euler("xyz", [20, -10, 0], degrees=True)
    rots = [r0 * Rotation.from_rotvec(w_body * ti) for ti in t]
    accel = np.array([r.inv().apply([0, 0, G]) for r in rots])
    gyro = np.tile(w_body, (len(t), 1))
    return t, gyro, accel, rots


def test_mahony_tracks_known_rotation():
    t, gyro, accel, rots = _synthetic()
    q = orientation.mahony(t, gyro, accel)
    est = orientation.to_scipy(q)
    truth = Rotation.concatenate(rots)
    # yaw is unobservable: compare the gravity direction (roll/pitch) only
    up_est = est.inv().apply([0, 0, 1])
    up_true = truth.inv().apply([0, 0, 1])
    err = np.degrees(np.arccos(np.clip((up_est * up_true).sum(1), -1, 1)))
    assert err[0] < 1.0  # init from 50 ms accel average while spinning
    assert err[5000:].max() < 0.1  # kp=1 -> ~1 s time constant; converged by 5 s


def test_gravity_corrects_gyro_bias_tilt():
    t, gyro, accel, _ = _synthetic()
    biased = gyro + np.array([0.01, -0.01, 0.0])  # ~0.6 deg/s bias
    q_free = orientation.mahony(t, biased, accel, kp=0.0, ki=0.0)
    q_corr = orientation.mahony(t, biased, accel)
    up = lambda q: orientation.to_scipy(q).inv().apply([0, 0, 1])
    truth = accel / np.linalg.norm(accel, axis=1, keepdims=True)
    e_free = np.degrees(np.arccos(np.clip((up(q_free) * truth).sum(1), -1, 1)))[-1]
    e_corr = np.degrees(np.arccos(np.clip((up(q_corr) * truth).sum(1), -1, 1)))[-1]
    assert e_corr < 1.0 < e_free


def test_resample_endpoints():
    t, gyro, accel, _ = _synthetic(dur=2.0)
    q = orientation.mahony(t, gyro, accel)
    r = orientation.resample(t, q, np.array([t[0], t[-1]]))
    assert np.allclose(np.abs((r * q[[0, -1]]).sum(1)), 1, atol=1e-9)
