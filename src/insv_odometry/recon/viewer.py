"""Side-by-side viewer: levelled 360 video + 3D sparse cloud with the moving camera.

Writes `<work>/viewer/` with index.html, video.mp4, cloud.bin and poses.json. Serve that
folder over HTTP (browsers block fetch() from file://), e.g.

    uv run insv-odometry serve out/<stem>/recon/viewer
"""

from __future__ import annotations

import json
from fractions import Fraction
from importlib import resources
from pathlib import Path

import av
import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

from .. import orientation
from ..parse import Telemetry
from ..video import read_tracks
from .lens import FisheyeLens
from .views import iter_frame_pairs


def frame_poses(work: Path, tel: Telemetry, t_frames: np.ndarray):
    """Camera(lens0)->world poses at every video frame.

    Positions: cubic spline through the SfM keyframes. Rotations: SfM keyframe rotation
    propagated with the gyro (via the estimated camera->IMU rotation and time offset), blended
    between the two neighbouring keyframes so there is no jump.
    """
    fusion = json.loads((work / "fusion.json").read_text())
    tum = np.loadtxt(work / "trajectory_metric_tum.txt")
    tk, pk, qk = tum[:, 0], tum[:, 1:4], tum[:, 4:8]
    Rk = Rotation.from_quat(qk)
    R_ic = Rotation.from_quat(fusion["R_cam_to_imu"]["quat_xyzw"])
    dt = fusion["time_offset_s"]

    q_gyro = orientation.mahony(tel.t, tel.gyro, tel.accel, kp=0.0, ki=0.0)
    gyro_at = lambda t: orientation.to_scipy(orientation.resample(tel.t, q_gyro, t + dt))
    Ri_k, Ri_f = gyro_at(tk), gyro_at(t_frames)

    tc = np.clip(t_frames, tk[0], tk[-1])
    pos = CubicSpline(tk, pk, axis=0)(tc)
    k0 = np.clip(np.searchsorted(tk, tc, side="right") - 1, 0, len(tk) - 2)
    k1 = k0 + 1
    frac = np.clip((tc - tk[k0]) / (tk[k1] - tk[k0]), 0, 1)

    def propagate(k):  # R_c(t) = R_c(tk) R_ic^T R_i(tk)^T R_i(t) R_ic
        return Rk[k] * R_ic.inv() * Ri_k[k].inv() * Ri_f * R_ic

    A, B = propagate(k0).as_quat(), propagate(k1).as_quat()
    B *= np.sign(np.sum(A * B, axis=1, keepdims=True))
    q = np.array([Slerp([0, 1], Rotation.from_quat([a, b]))(f).as_quat()
                  for a, b, f in zip(A, B, frac)])
    return pos, q, tk


def ground_height(pts: np.ndarray, traj: np.ndarray) -> float:
    d = np.min(np.linalg.norm(pts[:, None, :2] - traj[None, ::4, :2], axis=2), axis=1)
    z = pts[d < 3, 2]
    if len(z) < 100:
        return float(traj[:, 2].min() - 1.2)
    h, e = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.02, 0.02))
    return float(e[np.argmax(h)] + 0.01)


def render_video(src: Path, lenses: list[FisheyeLens], R_wc: np.ndarray, yaw0: float,
                 out: Path, width: int = 1920, fps: Fraction = Fraction(30000, 1001), crf: int = 22,
                 downscale: int = 2) -> None:
    """World-locked, horizon-level equirectangular video. Longitude 0 looks along `yaw0`."""
    h = width // 2
    lon = (np.arange(width) + 0.5) / width * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(h) + 0.5) / h * np.pi
    lon, lat = np.meshgrid(lon, lat)
    az = yaw0 - lon  # increasing longitude turns right (clockwise seen from above)
    d_world = np.stack([np.cos(lat) * np.cos(az), np.cos(lat) * np.sin(az), np.sin(lat)], -1)
    d_world = d_world.reshape(-1, 3).astype(np.float64)
    axes = np.stack([L.rig_from_lens_R[:, 2] for L in lenses])
    ds = float(downscale)

    container = av.open(str(out), "w", options={"movflags": "+faststart"})
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = width, h, "yuv420p"
    stream.options = {"crf": str(crf), "preset": "medium"}
    tracks = (lenses[0].track, lenses[1].track)
    n = len(R_wc)
    for idx, _, imgs in iter_frame_pairs(src, tracks):
        if idx >= n:
            break
        rays_rig = d_world @ R_wc[idx]  # R_wc^T d
        owner = np.argmax(rays_rig @ axes.T, axis=1)
        frame = np.zeros((h * width, 3), np.uint8)
        for li, L in enumerate(lenses):
            sel = np.nonzero(owner == li)[0]
            uv, valid = L.project(rays_rig[sel] @ L.rig_from_lens_R)
            small = cv2.resize(imgs[li], None, fx=1 / ds, fy=1 / ds, interpolation=cv2.INTER_AREA)
            m = ((uv + 0.5) / ds - 0.5).astype(np.float32)
            pad = (-len(m)) % 1024  # remap maps must stay below 32767 px per side
            m = np.concatenate([m, np.full((pad, 2), -1, np.float32)]).reshape(-1, 1024, 2)
            px = cv2.remap(small, m[..., 0], m[..., 1], cv2.INTER_LINEAR)
            px = px.reshape(-1, 3)[: len(sel)]
            px[~valid] = 0
            frame[sel] = px
        vf = av.VideoFrame.from_ndarray(frame.reshape(h, width, 3), format="bgr24")
        for pkt in stream.encode(vf):
            container.mux(pkt)
        print(f"\r  video frame {idx + 1}/{n}", end="", flush=True)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    print()


def build(work: Path, src: Path, tel: Telemetry, lenses: list[FisheyeLens],
          width: int = 1920, max_points: int = 200_000) -> Path:
    out = work / "viewer"
    out.mkdir(exist_ok=True)
    t_frames = read_tracks(src)[0].frame_times
    pos, q, tk = frame_poses(work, tel, t_frames)
    R_wc = Rotation.from_quat(q).as_matrix()

    # point cloud (metric, gravity aligned) written by the fuse stage
    raw = (work / "sparse_points_metric.ply").read_bytes()
    body = raw[raw.index(b"end_header\n") + len(b"end_header\n"):]
    pc = np.frombuffer(body, dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    pts, cols = pc["p"], pc["c"]
    centre = np.median(pos, axis=0)
    keep = np.linalg.norm(pts[:, :2] - centre[:2], axis=1) < 25
    pts, cols = pts[keep][:max_points], cols[keep][:max_points]
    ground = ground_height(pts, pos)

    travel = pos[-1, :2] - pos[0, :2]
    yaw0 = float(np.arctan2(travel[1], travel[0]))  # video centre = overall walking direction

    (out / "cloud.bin").write_bytes(pts.astype("<f4").tobytes() + cols.astype("u1").tobytes())
    fusion = json.loads((work / "fusion.json").read_text())
    (out / "poses.json").write_text(json.dumps({
        "fps": 30000 / 1001,
        "num_points": int(len(pts)),
        "ground_z": ground,
        "video_centre_yaw": yaw0,
        "keyframe_times": tk.round(4).tolist(),
        "t": t_frames.round(4).tolist(),
        "p": pos.round(4).tolist(),
        "q": q.round(6).tolist(),
        "info": {
            "camera": f"{tel.camera} {tel.model}",
            "scale": fusion["scale"]["value"],
            "path_length_m": fusion["path_length_m"],
            "registered_frames": fusion["registered_frames"],
            "time_offset_ms": fusion["time_offset_s"] * 1000,
        },
    }))
    html = resources.files(__package__).joinpath("viewer.html").read_text(encoding="utf-8")
    (out / "index.html").write_text(html, encoding="utf-8")

    print(f"Viewer: rendering levelled 360 video ({width}x{width // 2})")
    render_video(src, lenses, R_wc, yaw0, out / "video.mp4", width=width)
    print(f"  wrote {out}")
    return out
