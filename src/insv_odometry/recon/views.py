"""Render dual-fisheye video frames into a rig of virtual pinhole views (+ masks).

Each lens gets 5 virtual cameras (centre + 4 tilted). They share the lens optical
centre, so the rig is exact apart from lens calibration error. Masks give every
scene direction to exactly one view: the lens whose axis is closer, then that
lens's nearest view.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .lens import FisheyeLens


@dataclass
class VirtualView:
    name: str  # image folder / rig prefix, e.g. "lens0_c"
    lens: int
    cam_from_lens: np.ndarray  # 3x3
    size: int
    focal: float
    map_x: np.ndarray | None = None
    map_y: np.ndarray | None = None
    mask: np.ndarray | None = None


def _cam_rays(size: int, focal: float) -> np.ndarray:
    """(size, size, 3) unit rays for COLMAP pixel centres (i + 0.5)."""
    c = size / 2
    u = (np.arange(size) + 0.5 - c) / focal
    xx, yy = np.meshgrid(u, u)
    rays = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def view_rotations(tilt_deg: float) -> dict[str, np.ndarray]:
    """cam_from_lens rotations: centre, left/right (about y), up/down (about x)."""
    t = tilt_deg
    return {
        "c": np.eye(3),
        "l": Rotation.from_euler("y", t, degrees=True).as_matrix(),
        "r": Rotation.from_euler("y", -t, degrees=True).as_matrix(),
        "u": Rotation.from_euler("x", -t, degrees=True).as_matrix(),
        "d": Rotation.from_euler("x", t, degrees=True).as_matrix(),
    }


def build_views(lenses: list[FisheyeLens], size: int = 768, fov_deg: float = 90.0,
                tilt_deg: float = 50.0, edge_margin_deg: float = 3.0) -> list[VirtualView]:
    focal = size / 2 / np.tan(np.deg2rad(fov_deg) / 2)
    rays_cam = _cam_rays(size, focal)
    rots = view_rotations(tilt_deg)
    axes_rig = [L.rig_from_lens_R[:, 2] for L in lenses]  # lens optical axes in rig frame

    views = []
    for li, L in enumerate(lenses):
        centres = np.stack([R.T[:, 2] for R in rots.values()])  # view axes in lens frame
        for vname, R in rots.items():
            rays_lens = rays_cam @ R  # d_lens = R^T d_cam, row-vector form
            uv, valid = L.project(rays_lens)
            theta = np.arccos(np.clip(rays_lens[..., 2], -1, 1))
            valid &= theta <= L.fov_limit - np.deg2rad(edge_margin_deg)
            rays_rig = rays_lens @ L.rig_from_lens_R.T
            own_lens = np.argmax(np.stack([rays_rig @ a for a in axes_rig], -1), -1) == li
            own_view = np.argmax(rays_lens @ centres.T, -1) == list(rots).index(vname)
            mask = (valid & own_lens & own_view).astype(np.uint8) * 255
            views.append(VirtualView(
                name=f"lens{li}_{vname}", lens=li, cam_from_lens=R, size=size, focal=focal,
                map_x=uv[..., 0].astype(np.float32), map_y=uv[..., 1].astype(np.float32),
                mask=mask,
            ))
    return views


def cam_from_rig(view: VirtualView, lens: FisheyeLens) -> tuple[np.ndarray, np.ndarray]:
    """Rigid transform rig -> virtual camera (R, t)."""
    lens_from_rig_R = lens.rig_from_lens_R.T
    lens_from_rig_t = -lens_from_rig_R @ lens.rig_from_lens_t
    R = view.cam_from_lens @ lens_from_rig_R
    t = view.cam_from_lens @ lens_from_rig_t
    return R, t


def iter_frame_pairs(path: str | Path, tracks: tuple[int, int], step: int = 1,
                     start: int = 0, stop: int | None = None):
    """Yield (frame_index, pts_s, [img_track_a, img_track_b]) decoding both tracks in lockstep."""
    conts = [av.open(str(path)) for _ in tracks]
    try:
        gens = []
        for c, ti in zip(conts, tracks):
            s = c.streams.video[ti] if ti < len(c.streams.video) else c.streams[ti]
            s.thread_type = "AUTO"
            gens.append(c.decode(s))
        for i, frames in enumerate(zip(*gens)):
            if stop is not None and i >= stop:
                break
            if i < start or (i - start) % step:
                continue
            t = float(frames[0].pts * frames[0].time_base) if frames[0].pts is not None else np.nan
            yield i, t, [f.to_ndarray(format="bgr24") for f in frames]
    finally:
        for c in conts:
            c.close()


def equirect(images: list[np.ndarray], lenses: list[FisheyeLens], width: int = 2048) -> np.ndarray:
    """Stitch both lenses to an equirectangular image in the rig frame (seam check)."""
    h = width // 2
    lon = (np.arange(width) + 0.5) / width * 2 * np.pi - np.pi
    lat = (np.arange(h) + 0.5) / h * np.pi - np.pi / 2
    lon, lat = np.meshgrid(lon, lat)
    rays_rig = np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)], -1)
    axes = np.stack([L.rig_from_lens_R[:, 2] for L in lenses])
    owner = np.argmax(rays_rig @ axes.T, -1)
    out = np.zeros((h, width, 3), np.uint8)
    for li, L in enumerate(lenses):
        rays_lens = rays_rig @ L.rig_from_lens_R  # R^T d
        uv, valid = L.project(rays_lens)
        img = cv2.remap(images[li], uv[..., 0].astype(np.float32), uv[..., 1].astype(np.float32),
                        cv2.INTER_LINEAR)
        sel = valid & (owner == li)
        out[sel] = img[sel]
    # draw the seam
    seam = np.abs(np.diff(owner, axis=1, prepend=owner[:, :1])) > 0
    out[seam] = (0, 0, 255)
    return out


def render(path: str | Path, lenses: list[FisheyeLens], views: list[VirtualView],
           out_dir: Path, step: int, quality: int = 95, check_every: int = 0,
           downscale: int = 2, max_frames: int | None = None) -> list[dict]:
    """Write images/<view>/frame_XXXXX.jpg and masks; the fisheye is first area-downscaled
    by `downscale` so the bilinear remap does not alias."""
    img_dir, mask_dir = out_dir / "images", out_dir / "masks"
    for v in views:
        (img_dir / v.name).mkdir(parents=True, exist_ok=True)
        (mask_dir / v.name).mkdir(parents=True, exist_ok=True)
    tracks = (lenses[0].track, lenses[1].track)
    ds = float(downscale)
    maps = {v.name: ((v.map_x + 0.5) / ds - 0.5, (v.map_y + 0.5) / ds - 0.5) for v in views}
    rows = []
    stop = None if max_frames is None else max_frames * step
    for idx, t, imgs in iter_frame_pairs(path, tracks, step=step, stop=stop):
        fname = f"frame_{idx:05d}.jpg"
        small = [cv2.resize(im, None, fx=1 / ds, fy=1 / ds, interpolation=cv2.INTER_AREA)
                 if downscale > 1 else im for im in imgs]
        for v in views:
            mx, my = maps[v.name]
            im = cv2.remap(small[v.lens], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            im[v.mask == 0] = 0
            cv2.imwrite(str(img_dir / v.name / fname), im, [cv2.IMWRITE_JPEG_QUALITY, quality])
            mpath = mask_dir / v.name / f"{fname}.png"
            if not mpath.exists():
                cv2.imwrite(str(mpath), v.mask)
        if check_every and len(rows) % check_every == 0:
            cv2.imwrite(str(out_dir / f"equirect_{idx:05d}.jpg"), equirect(imgs, lenses))
        rows.append({"frame": idx, "t_s": t, "image": fname})
        print(f"\r  rendered frame {idx}", end="", flush=True)
    print()
    with (out_dir / "frames.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "t_s", "image"])
        w.writeheader()
        w.writerows(rows)
    return rows
