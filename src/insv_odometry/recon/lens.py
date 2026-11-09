"""Insta360 dual-fisheye lens model (Mei unified model + radtan) and rig extrinsics.

Intrinsics come from the `offset_v3` metadata block (k1, k2, k3, p1, p2, xi, f, c),
which is in sensor coordinates (7680 px per lens, before the window crop).
They are rescaled to the recorded track resolution.
Lens-to-lens extrinsics come from telemetry-parser's rig.

Camera frame convention (OpenCV / COLMAP): x right, y down, z forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

_V3_BLOCK = 19  # values per lens in offset_v3


@dataclass
class FisheyeLens:
    name: str
    track: int  # video stream index in the .insv
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    xi: float
    k: np.ndarray  # k1, k2, k3
    p: np.ndarray  # p1, p2
    fov_limit: float  # rad from optical axis
    rig_from_lens_R: np.ndarray  # 3x3
    rig_from_lens_t: np.ndarray  # 3, metres

    def project(self, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project (..., 3) rays in lens frame to (..., 2) pixels. Returns (uv, valid)."""
        d = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
        theta = np.arccos(np.clip(d[..., 2], -1.0, 1.0))
        z = d[..., 2] + self.xi
        valid = (theta <= self.fov_limit) & (z > 1e-6)
        z = np.where(valid, z, 1.0)
        mx, my = d[..., 0] / z, d[..., 1] / z
        r2 = mx * mx + my * my
        k1, k2, k3 = self.k
        p1, p2 = self.p
        radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        xd = mx * radial + 2 * p1 * mx * my + p2 * (r2 + 2 * mx * mx)
        yd = my * radial + p1 * (r2 + 2 * my * my) + 2 * p2 * mx * my
        uv = np.stack([self.fx * xd + self.cx, self.fy * yd + self.cy], axis=-1)
        valid &= (uv[..., 0] >= 0) & (uv[..., 0] <= self.width - 1)
        valid &= (uv[..., 1] >= 0) & (uv[..., 1] <= self.height - 1)
        return uv, valid

    def image_circle_radius_px(self) -> float:
        """Radius in pixels of the ray at the FOV limit (sanity check vs metadata)."""
        ray = np.array([np.sin(self.fov_limit), 0.0, np.cos(self.fov_limit)])
        uv, _ = self.project(ray[None])
        return float(uv[0, 0] - self.cx)


def _quat(q: dict[str, float]) -> np.ndarray:
    return Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()


def load_lenses(metadata: dict[str, Any], rig: dict[str, Any],
                track_size: tuple[int, int]) -> list[FisheyeLens]:
    """Build both lenses. Lens 0 is the rig reference (its frame = rig frame)."""
    v3 = metadata.get("offset_v3") or metadata.get("original_offset_v3")
    if not v3 or len(v3) < 1 + 2 * _V3_BLOCK:
        raise ValueError("offset_v3 lens calibration not found in metadata")
    crop = metadata.get("window_crop_info") or {}
    src_w = float(crop.get("src_width", 2 * track_size[0]))
    dst_w = float(crop.get("dst_width", src_w))
    crop_px = (src_w - dst_w) / 2 + float(crop.get("crop_offset_x", 0))
    scale = track_size[0] / dst_w

    cams = rig["projection"]["kind"]["Rig"]["cameras"]
    pictures = rig.get("pictures", [])

    # Rig frame := lens 0 frame (telemetry-parser gives a near-identity rotation for lens 0).
    R0 = _quat(cams[0]["rotation"])
    t0 = np.asarray(cams[0]["translation"], dtype=float)

    lenses = []
    for i, cam in enumerate(cams[:2]):
        b = v3[1 + i * _V3_BLOCK : 1 + (i + 1) * _V3_BLOCK]
        xi, fx, fy, cx, cy = b[0:5]
        k = np.array(b[11:14])
        p = np.array(b[14:16])
        cx -= i * src_w  # lens 1 sits in the right half of the side-by-side sensor layout
        pic = pictures[cam["picture"]["input"]] if pictures else {"VideoTrack": i}
        Ri = _quat(cam["rotation"])
        ti = np.asarray(cam["translation"], dtype=float)
        lenses.append(
            FisheyeLens(
                name=f"lens{i}",
                track=int(pic.get("VideoTrack", i)),
                width=track_size[0],
                height=track_size[1],
                fx=fx * scale,
                fy=fy * scale,
                cx=(cx - crop_px) * scale,
                cy=(cy - crop_px) * scale,
                xi=xi,
                k=k,
                p=p,
                fov_limit=float(cam.get("fov_limit", np.deg2rad(100))),
                rig_from_lens_R=R0.T @ Ri,
                rig_from_lens_t=R0.T @ (ti - t0),
            )
        )
    return lenses
