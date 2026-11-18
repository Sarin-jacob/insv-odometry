"""Flat (rectilinear) video from the dual fisheye, aimed at a subject that the camera circles.

The view follows the horizontal heading of a chosen lens axis (smoothed), keeps the horizon
level using the IMU gravity direction, and adds a fixed tilt. This suits footage where the
camera orbits a subject while one lens keeps facing it.

Also exports the camera's IMU samples as text, exactly as telemetry-parser decodes them.
"""

from __future__ import annotations

import csv
import json
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import telemetry_parser
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation

from .. import orientation
from ..parse import Telemetry
from ..video import read_tracks
from .lens import FisheyeLens
from .views import iter_frame_pairs

# Camera(lens 0)->IMU rotation of the X4 Air, estimated on the well-conditioned walking clip
# (0.12 deg per-step agreement). Used when no fusion.json for the clip is given.
X4_AIR_CAM_TO_IMU_XYZW = [0.7088031700620934, -0.006066673204794725,
                          0.7053782785007235, 0.001657047226055434]


def view_bases(up_rig: np.ndarray, axis_rig: np.ndarray, t: np.ndarray, tilt_deg: float,
               smooth_s: float) -> np.ndarray:
    """Per-frame rig->view rotation rows (x right, y down, z forward), shape (n, 3, 3)."""
    h = axis_rig - np.einsum("ij,ij->i", up_rig, axis_rig)[:, None] * up_rig  # on horizontal
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    fps = 1.0 / np.median(np.diff(t))
    win = max(1, int(round(smooth_s * fps)))
    h = uniform_filter1d(h, size=win, axis=0, mode="nearest")
    h -= (np.einsum("ij,ij->i", h, up_rig))[:, None] * up_rig
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    el = np.radians(tilt_deg)
    z = np.cos(el) * h + np.sin(el) * up_rig
    down = -up_rig
    y = down - np.einsum("ij,ij->i", down, z)[:, None] * z
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=1)  # rows: view axes expressed in rig frame


def render_flat(src: Path, lenses: list[FisheyeLens], bases: np.ndarray, out: Path, start: int,
                width: int, height: int, vfov_deg: float, crf: int = 18) -> None:
    f = (height / 2) / np.tan(np.radians(vfov_deg) / 2)
    u, v = np.meshgrid(np.arange(width) + 0.5 - width / 2, np.arange(height) + 0.5 - height / 2)
    rays_view = np.stack([u, v, np.full_like(u, f)], -1).reshape(-1, 3)
    rays_view /= np.linalg.norm(rays_view, axis=1, keepdims=True)
    axes = np.stack([L.rig_from_lens_R[:, 2] for L in lenses])

    container = av.open(str(out), "w", options={"movflags": "+faststart"})
    stream = container.add_stream("libx264", rate=Fraction(30000, 1001))
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    stream.options = {"crf": str(crf), "preset": "medium"}
    n = len(bases)
    for idx, _, imgs in iter_frame_pairs(src, (lenses[0].track, lenses[1].track),
                                         start=start, stop=start + n):
        k = idx - start
        rays_rig = rays_view @ bases[k]  # sum_i ray_i * axis_i
        owner = np.argmax(rays_rig @ axes.T, axis=1)
        frame = np.zeros((width * height, 3), np.uint8)
        for li, L in enumerate(lenses):
            sel = np.nonzero(owner == li)[0]
            if not len(sel):
                continue
            uv, ok = L.project(rays_rig[sel] @ L.rig_from_lens_R)
            m = uv.astype(np.float32)
            pad = (-len(m)) % 1024  # cv2.remap maps must stay below 32767 px per side
            m = np.concatenate([m, np.full((pad, 2), -1, np.float32)]).reshape(-1, 1024, 2)
            px = cv2.remap(imgs[li], m[..., 0], m[..., 1], cv2.INTER_LINEAR).reshape(-1, 3)[: len(sel)]
            px[~ok] = 0
            frame[sel] = px
        vf = av.VideoFrame.from_ndarray(frame.reshape(height, width, 3), format="bgr24")
        for pkt in stream.encode(vf):
            container.mux(pkt)
        print(f"\r  flat frame {k + 1}/{n}", end="", flush=True)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    print()


def export_raw_imu(src: Path, out: Path, t_video0: float) -> int:
    """Camera IMU samples as decoded by telemetry-parser: no bias removal, filtering, rotation
    or resampling. Values are divided by the stored Scale to give deg/s and g."""
    tp = telemetry_parser.Parser(str(src))
    gyro = acc = None
    gs = as_ = 1.0
    for grp in tp.telemetry():
        if "Gyroscope" in grp:
            gyro, gs = grp["Gyroscope"]["Data"], float(grp["Gyroscope"].get("Scale", 1.0))
        if "Accelerometer" in grp:
            acc, as_ = grp["Accelerometer"]["Data"], float(grp["Accelerometer"].get("Scale", 1.0))
    if gyro is None or acc is None or len(gyro) != len(acc):
        raise ValueError("gyro/accel streams missing or of different length")
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_clip_s", "t_flatvideo_s", "gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
                    "accel_x_g", "accel_y_g", "accel_z_g"])
        for g, a in zip(gyro, acc):
            w.writerow([f"{g['t']:.6f}", f"{g['t'] - t_video0:.6f}",
                        f"{g['x'] / gs:.6f}", f"{g['y'] / gs:.6f}", f"{g['z'] / gs:.6f}",
                        f"{a['x'] / as_:.7f}", f"{a['y'] / as_:.7f}", f"{a['z'] / as_:.7f}"])
    return len(gyro)


def export_frames(src: Path, out: Path, frame_times: np.ndarray, start: int, stop: int) -> None:
    tp = telemetry_parser.Parser(str(src))
    exp = np.array([[e["t"], e["v"]] for grp in tp.telemetry() if "Exposure" in grp
                    for e in grp["Exposure"]["Data"]])
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["flat_frame", "clip_frame", "t_clip_s", "t_flatvideo_s", "exposure_s"])
        t0 = frame_times[start]
        for k, i in enumerate(range(start, stop)):
            j = int(np.argmin(np.abs(exp[:, 0] - frame_times[i]))) if len(exp) else -1
            w.writerow([k, i, f"{frame_times[i]:.6f}", f"{frame_times[i] - t0:.6f}",
                        f"{exp[j, 1]:.9f}" if j >= 0 else ""])


def build(src: Path, tel: Telemetry, lenses: list[FisheyeLens], out_dir: Path, start_s: float,
          end_s: float | None, lens_index: int = 1, tilt_deg: float = 12.0, vfov_deg: float = 105.0,
          width: int = 1440, height: int = 1080, smooth_s: float = 1.0,
          fusion_json: Path | None = None, name: str | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ft = read_tracks(src)[0].frame_times
    i0 = int(np.searchsorted(ft, start_s - 1e-6))
    i1 = len(ft) if end_s is None else int(np.searchsorted(ft, end_s + 1e-6))
    t = ft[i0:i1]

    q_xyzw, dt, src_rot = X4_AIR_CAM_TO_IMU_XYZW, 0.0, "X4 Air default (walking clip)"
    if fusion_json and Path(fusion_json).exists():
        fj = json.loads(Path(fusion_json).read_text())
        dt = float(fj.get("time_offset_s", 0.0))
    R_ic = Rotation.from_quat(q_xyzw).as_matrix()
    q_imu = orientation.mahony(tel.t, tel.gyro, tel.accel)  # gravity-corrected, IMU->world
    R_gi = orientation.to_scipy(orientation.resample(tel.t, q_imu, t + dt)).as_matrix()
    R_gc = R_gi @ R_ic  # camera(rig)->world
    up_rig = np.einsum("nji,j->ni", R_gc, np.array([0.0, 0.0, 1.0]))  # world up in rig frame
    axis = np.tile(lenses[lens_index].rig_from_lens_R[:, 2], (len(t), 1))
    bases = view_bases(up_rig, axis, t, tilt_deg, smooth_s)

    stem = name or src.stem
    tag = f"{start_s:g}-{end_s:g}s" if end_s is not None else f"{start_s:g}s-end"
    video = out_dir / f"{stem}_flat_{tag}.mp4"
    print(f"Flat video {width}x{height}, vfov {vfov_deg} deg, tilt {tilt_deg} deg, "
          f"frames {i0}-{i1 - 1} -> {video}")
    render_flat(src, lenses, bases, video, i0, width, height, vfov_deg)

    imu_csv = out_dir / f"{stem}_imu_raw.csv"
    n_imu = export_raw_imu(src, imu_csv, float(ft[i0]))
    frames_csv = out_dir / f"{stem}_frames_{tag}.csv"
    export_frames(src, frames_csv, ft, i0, i1)
    return {"video": video, "imu_csv": imu_csv, "frames_csv": frames_csv, "imu_samples": n_imu,
            "frames": i1 - i0, "R_cam_to_imu_source": src_rot, "time_offset_s": dt}


def add_parser(sub) -> None:
    r = sub.add_parser("flat", help="flat video aimed along one lens's heading + raw IMU text export")
    r.add_argument("file", help="path to .insv")
    r.add_argument("-o", "--out", required=True, help="output folder")
    r.add_argument("--start", type=float, default=0.0, help="window start (s, video time)")
    r.add_argument("--end", type=float, default=None, help="window end (s, video time)")
    r.add_argument("--lens", type=int, default=1, help="lens whose heading to follow (0 or 1)")
    r.add_argument("--tilt", type=float, default=8.0, help="view tilt above horizontal (deg)")
    r.add_argument("--vfov", type=float, default=110.0, help="vertical field of view (deg)")
    r.add_argument("--size", default="1440x1080", help="WIDTHxHEIGHT")
    r.add_argument("--smooth", type=float, default=1.0, help="heading smoothing window (s)")
    r.add_argument("--fusion", help="fusion.json of a recon run (uses its IMU time offset)")
    r.set_defaults(func=_run)


def _run(args) -> int:
    from .. import parse
    from . import lens as lens_mod

    src = Path(args.file)
    tel = parse.load(src)
    tr = read_tracks(src)[0]
    lenses = lens_mod.load_lenses(tel.metadata, tel.rig, (tr.width, tr.height))
    w, h = (int(v) for v in args.size.lower().split("x"))
    res = build(src, tel, lenses, Path(args.out), args.start, args.end, lens_index=args.lens,
                tilt_deg=args.tilt, vfov_deg=args.vfov, width=w, height=h, smooth_s=args.smooth,
                fusion_json=Path(args.fusion) if args.fusion else None)
    for k, v in res.items():
        print(f"  {k}: {v}")
    return 0
