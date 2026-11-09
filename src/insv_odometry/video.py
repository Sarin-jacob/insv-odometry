"""Video track info and frame timestamps (container only, no decoding)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np


@dataclass
class VideoTrack:
    index: int
    codec: str
    width: int
    height: int
    fps: float
    frame_times: np.ndarray  # seconds, presentation order


def read_tracks(path: str | Path) -> list[VideoTrack]:
    tracks: list[VideoTrack] = []
    with av.open(str(path)) as c:
        streams = list(c.streams.video)
        pts: dict[int, list[int]] = {s.index: [] for s in streams}
        for pkt in c.demux(*streams):
            if pkt.pts is not None and pkt.stream.index in pts:
                pts[pkt.stream.index].append(pkt.pts)
        for s in streams:
            p = np.sort(np.array(pts[s.index], dtype=np.int64))
            start = s.start_time or 0
            times = (p - start) * float(s.time_base)
            tracks.append(
                VideoTrack(
                    index=s.index,
                    codec=s.codec_context.name,
                    width=s.codec_context.width,
                    height=s.codec_context.height,
                    fps=float(s.average_rate) if s.average_rate else 0.0,
                    frame_times=times,
                )
            )
    return tracks
