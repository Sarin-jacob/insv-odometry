"""`insv-odometry recon`: render views -> rig SfM -> IMU fusion."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from .. import parse
from ..video import read_tracks
from . import lens as lens_mod
from . import views as views_mod


def add_parser(sub: argparse._SubParsersAction) -> None:
    r = sub.add_parser("recon", help="phase 2: SfM on virtual pinhole views, fused with IMU")
    r.add_argument("file", help="path to .insv")
    r.add_argument("-o", "--out", help="work folder (default: out/<stem>/recon)")
    r.add_argument("--step", type=int, default=6, help="use every Nth video frame (default 6 = 5 fps)")
    r.add_argument("--size", type=int, default=768, help="virtual view size in px")
    r.add_argument("--max-features", type=int, default=2048)
    r.add_argument("--overlap", type=int, default=6, help="sequential matching overlap (frames)")
    r.add_argument("--refine-rig", action="store_true", help="let BA refine lens-to-lens extrinsics")
    r.add_argument("--stages", default="render,sfm,fuse,viewer",
                   help="comma list of stages to run (render, sfm, fuse, viewer)")
    r.add_argument("--video-width", type=int, default=1920, help="viewer 360 video width")
    r.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    src = Path(args.file)
    work = Path(args.out) if args.out else Path("out") / src.stem / "recon"
    work.mkdir(parents=True, exist_ok=True)
    stages = {s.strip() for s in args.stages.split(",")}

    tel = parse.load(src)
    tracks = read_tracks(src)
    size = (tracks[0].width, tracks[0].height)
    lenses = lens_mod.load_lenses(tel.metadata, tel.rig, size)
    views = views_mod.build_views(lenses, size=args.size)
    print(f"{tel.camera} {tel.model}: {len(views)} virtual views of {args.size}px, "
          f"lens offset {lenses[1].rig_from_lens_t[2] * 100:.2f} cm")

    if "render" in stages:
        t0 = time.time()
        for d in ("images", "masks"):
            shutil.rmtree(work / d, ignore_errors=True)
        print(f"Rendering every {args.step}th frame -> {work}")
        rows = views_mod.render(src, lenses, views, work, step=args.step, check_every=25)
        print(f"  {len(rows)} frames x {len(views)} views in {time.time() - t0:.0f} s")

    rec = None
    if "sfm" in stages:
        from . import sfm

        print("SfM (pycolmap, CPU)")
        rec = sfm.run(work, views, lenses, max_features=args.max_features,
                      overlap=args.overlap, refine_rig=args.refine_rig)
        best = work / "sparse_best"
        best.mkdir(exist_ok=True)
        rec.write(best)
        rec.export_PLY(str(work / "sparse_points.ply"))
        print(f"  wrote {best} and sparse_points.ply")

    if "fuse" in stages:
        import pycolmap

        from . import fuse

        if rec is None:
            rec = pycolmap.Reconstruction(str(work / "sparse_best"))
        fuse.run(work, rec, tel, views, lenses)

    if "viewer" in stages:
        from . import viewer

        out = viewer.build(work, src, tel, lenses, width=args.video_width)
        print(f"View it: uv run insv-odometry serve \"{out}\"  -> http://127.0.0.1:8770/")
    return 0

