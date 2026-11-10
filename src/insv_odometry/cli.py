"""Command line: insv-odometry extract FILE -o OUT."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import export, orientation, parse, video


def extract(args: argparse.Namespace) -> int:
    src = Path(args.file)
    out = Path(args.out) if args.out else Path("out") / src.stem
    out.mkdir(parents=True, exist_ok=True)

    print(f"Reading {src}")
    tel = parse.load(src)
    print(f"  {tel.camera} {tel.model}: {len(tel.t)} IMU samples at {tel.imu_rate_hz:.0f} Hz, "
          f"t = {tel.t[0]:.3f} .. {tel.t[-1]:.3f} s")

    tracks = video.read_tracks(src)
    for tr in tracks:
        print(f"  video track {tr.index}: {tr.codec} {tr.width}x{tr.height} @ {tr.fps:.3f} fps, "
              f"{len(tr.frame_times)} frames, {tr.frame_times[0]:.3f} .. {tr.frame_times[-1]:.3f} s")

    bias = orientation.estimate_gyro_bias(tel.t, tel.gyro) if args.bias else np.zeros(3)
    q = orientation.mahony(tel.t, tel.gyro, tel.accel, kp=args.kp, ki=args.ki, bias=bias)
    q_gyro = orientation.mahony(tel.t, tel.gyro, tel.accel, kp=0.0, ki=0.0, bias=bias)

    export.write_imu(out, tel)
    export.write_orientation(out, tel.t, q, "orientation_imu.csv")

    frame_q = None
    if tracks:
        tr = tracks[0]  # both lenses are recorded in lockstep
        frame_q = orientation.resample(tel.t, q, tr.frame_times)
        if len(tel.exposure):
            exp = np.interp(tr.frame_times, tel.exposure[:, 0], tel.exposure[:, 1])
        else:
            exp = np.full(len(tr.frame_times), np.nan)
        export.write_frames(out, tr, frame_q, exp)
        export.write_tum(out, tr.frame_times, frame_q)

    export.write_json(out / "metadata.json", tel.metadata)
    export.write_json(out / "calibration.json", {
        "camera": tel.camera,
        "model": tel.model,
        "lens": tel.lens,
        "rig": tel.rig,
        "imu": {
            "rate_hz": tel.imu_rate_hz,
            "frame": "telemetry-parser normalized IMU frame",
            "gyro_bias_rad_s_removed": bias,
            "imu_turn_ypr_deg": tel.metadata.get("imu_turn_ypr")
            or tel.rig.get("vendor", {}).get("Insta360", {}).get("imu_turn_ypr"),
            "gyro_calib": tel.metadata.get("gyro_calib"),
        },
        "video_tracks": [
            {"index": t.index, "codec": t.codec, "width": t.width, "height": t.height,
             "fps": t.fps, "frames": len(t.frame_times)}
            for t in tracks
        ],
    })

    # Summary / sanity numbers
    acc_norm = np.linalg.norm(tel.accel, axis=1)
    total_rot = np.rad2deg(np.sum(np.linalg.norm(tel.gyro - bias, axis=1) * np.gradient(tel.t)))
    corr = export.rotation_diff_deg(q, q_gyro)
    summary = {
        "source": str(src),
        "camera": f"{tel.camera} {tel.model}",
        "imu_samples": len(tel.t),
        "imu_rate_hz": round(tel.imu_rate_hz, 2),
        "imu_t_range_s": [float(tel.t[0]), float(tel.t[-1])],
        "video_frames": len(tracks[0].frame_times) if tracks else 0,
        "imu_covers_all_frames": bool(tracks and tel.t[0] <= tracks[0].frame_times[0]
                                      and tel.t[-1] >= tracks[0].frame_times[-1]),
        "accel_norm_m_s2": {"mean": float(acc_norm.mean()), "std": float(acc_norm.std())},
        "gyro_bias_rad_s": bias,
        "total_rotation_deg": float(total_rot),
        "gravity_correction_deg": {"max": float(corr.max()), "end": float(corr[-1])},
        "final_orientation_deg_rpy": export.euler_deg(q[-1:])[0],
        "filter": {"kind": "mahony", "kp": args.kp, "ki": args.ki},
    }
    export.write_json(out / "summary.json", summary)
    if not args.no_plot:
        export.plot(out, tel, q, q_gyro)

    print(f"  total rotation {total_rot:.0f} deg, |accel| mean {acc_norm.mean():.2f} m/s^2")
    print(f"  gravity correction vs gyro-only: max {corr.max():.2f} deg, end {corr[-1]:.2f} deg")
    print(f"Wrote {out.resolve()}")
    for p in sorted(out.iterdir()):
        print(f"  {p.name:22s} {p.stat().st_size / 1024:9.1f} KiB")
    return 0


def _serve(args: argparse.Namespace) -> int:
    from .serve import serve

    d = Path(args.dir) if args.dir else None
    if d is None:
        found = sorted(Path("out").glob("*/recon*/viewer/index.html"), key=lambda p: p.stat().st_mtime)
        if not found:
            sys.exit("No viewer found under out/*/recon/viewer; run `recon` first or pass a folder.")
        d = found[-1].parent
    serve(d, port=args.port, host=args.host)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="insv-odometry", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="extract IMU, orientation, frame poses and calibration")
    e.add_argument("file", help="path to .insv (or Insta360 .mp4)")
    e.add_argument("-o", "--out", help="output folder (default: out/<file stem>)")
    e.add_argument("--kp", type=float, default=1.0, help="Mahony proportional gain")
    e.add_argument("--ki", type=float, default=0.01, help="Mahony integral gain")
    e.add_argument("--no-bias", dest="bias", action="store_false",
                   help="do not subtract gyro bias from the stillest window")
    e.add_argument("--no-plot", action="store_true")
    e.set_defaults(func=extract)

    s = sub.add_parser("serve", help="serve a viewer folder over HTTP (with video seeking)")
    s.add_argument("dir", nargs="?", help="folder to serve (default: newest out/*/recon*/viewer)")
    s.add_argument("--port", type=int, default=8770)
    s.add_argument("--host", default="127.0.0.1")
    s.set_defaults(func=_serve)

    try:
        from .recon import pipeline
    except ImportError:
        r = sub.add_parser("recon", help="phase 2 (needs: uv sync --extra recon)")
        r.set_defaults(func=lambda a: sys.exit("Install the recon extra: uv sync --extra recon"))
    else:
        pipeline.add_parser(sub)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
