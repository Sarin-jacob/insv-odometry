"""Read telemetry from an .insv file with telemetry-parser."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import telemetry_parser

G = 9.80665


@dataclass
class Telemetry:
    camera: str
    model: str
    t: np.ndarray  # (N,) seconds, 0 = first video frame
    gyro: np.ndarray  # (N, 3) rad/s, IMU frame
    accel: np.ndarray  # (N, 3) m/s^2, IMU frame (specific force: reads +g "up" at rest)
    exposure: np.ndarray  # (M, 2) [t_s, exposure_s]
    metadata: dict[str, Any] = field(default_factory=dict)
    lens: dict[str, Any] = field(default_factory=dict)
    rig: dict[str, Any] = field(default_factory=dict)

    @property
    def imu_rate_hz(self) -> float:
        return float(1.0 / np.median(np.diff(self.t)))


def _jsonable(obj: Any) -> Any:
    """Convert parser output (tuples, bytes, numpy) to plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (bytes, bytearray)):
        return f"<{len(obj)} bytes>"
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def load(path: str | Path) -> Telemetry:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    parser = telemetry_parser.Parser(str(path))
    imu = parser.normalized_imu()
    if not imu:
        raise ValueError(f"No IMU data in {path}")

    t = np.array([s["timestamp_ms"] for s in imu], dtype=np.float64) / 1000.0
    # normalized_imu: gyro in deg/s, accel in m/s^2, already rotated to the
    # parser's normalized IMU frame.
    gyro = np.deg2rad(np.array([s["gyro"] for s in imu], dtype=np.float64))
    accel = np.array([s["accl"] for s in imu], dtype=np.float64)

    order = np.argsort(t, kind="stable")
    t, gyro, accel = t[order], gyro[order], accel[order]
    keep = np.concatenate([[True], np.diff(t) > 0])  # drop duplicate stamps
    t, gyro, accel = t[keep], gyro[keep], accel[keep]

    metadata: dict[str, Any] = {}
    lens: dict[str, Any] = {}
    rig: dict[str, Any] = {}
    exposure = np.zeros((0, 2))
    for group in parser.telemetry():
        default = group.get("Default", {})
        if "Metadata" in default:
            metadata = _jsonable(default["Metadata"])
        lens_group = group.get("Lens", {})
        if "Data" in lens_group:
            lens = _jsonable(lens_group["Data"])
        if "Rig" in lens_group:
            rig = _jsonable(lens_group["Rig"])
        exp = group.get("Exposure", {}).get("Data")
        if exp:
            exposure = np.array([[e["t"], e["v"]] for e in exp], dtype=np.float64)

    return Telemetry(
        camera=parser.camera or "",
        model=parser.model or "",
        t=t,
        gyro=gyro,
        accel=accel,
        exposure=exposure,
        metadata=metadata,
        lens=lens,
        rig=rig,
    )
