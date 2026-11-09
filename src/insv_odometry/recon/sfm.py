"""Rig structure-from-motion with pycolmap (CPU)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pycolmap

from .lens import FisheyeLens
from .views import VirtualView, cam_from_rig


def rig_config(views: list[VirtualView], lenses: list[FisheyeLens]) -> pycolmap.RigConfig:
    cams = []
    for i, v in enumerate(views):
        c = v.size / 2
        camera = pycolmap.Camera(model="PINHOLE", width=v.size, height=v.size,
                                 params=[v.focal, v.focal, c, c])
        R, t = cam_from_rig(v, lenses[v.lens])
        cams.append(pycolmap.RigConfigCamera(
            ref_sensor=i == 0,
            image_prefix=f"{v.name}/",
            cam_from_rig=None if i == 0 else pycolmap.Rigid3d(
                pycolmap.Rotation3d(R), t.reshape(3, 1)),
            camera=camera,
        ))
    return pycolmap.RigConfig(cameras=cams)


def run(work: Path, views: list[VirtualView], lenses: list[FisheyeLens],
        max_features: int = 2048, overlap: int = 6, refine_rig: bool = False,
        num_threads: int = -1) -> pycolmap.Reconstruction:
    assert np.allclose(cam_from_rig(views[0], lenses[views[0].lens])[0], np.eye(3)), \
        "first view must be the rig reference"
    image_dir, mask_dir = work / "images", work / "masks"
    db_path = work / "database.db"
    sparse = work / "sparse"
    if db_path.exists():
        db_path.unlink()
    sparse.mkdir(parents=True, exist_ok=True)
    v0 = views[0]
    c = v0.size / 2

    t0 = time.time()
    ext = pycolmap.FeatureExtractionOptions(use_gpu=False, num_threads=num_threads)
    ext.sift.max_num_features = max_features
    pycolmap.extract_features(
        db_path, image_dir,
        reader_options=pycolmap.ImageReaderOptions(
            mask_path=mask_dir, camera_model="PINHOLE",
            camera_params=f"{v0.focal},{v0.focal},{c},{c}"),
        camera_mode=pycolmap.CameraMode.PER_FOLDER,
        extraction_options=ext,
    )
    print(f"  features: {time.time() - t0:.0f} s")

    with pycolmap.Database.open(db_path) as db:
        pycolmap.apply_rig_config([rig_config(views, lenses)], db)

    t0 = time.time()
    mopt = pycolmap.FeatureMatchingOptions(use_gpu=False, num_threads=num_threads)
    mopt.rig_verification = True
    mopt.skip_image_pairs_in_same_frame = True
    vopt = pycolmap.TwoViewGeometryOptions()
    vopt.filter_stationary_matches = True  # operator / camera body move with the rig
    pycolmap.match_sequential(
        db_path, matching_options=mopt,
        pairing_options=pycolmap.SequentialPairingOptions(overlap=overlap, quadratic_overlap=True),
        verification_options=vopt,
    )
    print(f"  matching: {time.time() - t0:.0f} s")

    t0 = time.time()
    opts = pycolmap.IncrementalPipelineOptions(
        num_threads=num_threads,
        ba_refine_sensor_from_rig=refine_rig,
        ba_refine_focal_length=False,
        ba_refine_principal_point=False,
        ba_refine_extra_params=False,
    )
    recs = pycolmap.incremental_mapping(db_path, image_dir, sparse, opts)
    print(f"  mapping: {time.time() - t0:.0f} s, {len(recs)} model(s)")
    if not recs:
        raise RuntimeError("SfM produced no reconstruction")
    for i, r in recs.items():
        print(f"   model {i}: {r.num_reg_frames()} frames, {r.num_reg_images()} images, "
              f"{r.num_points3D()} points, reproj {r.compute_mean_reprojection_error():.2f} px")
    best = max(recs.values(), key=lambda r: r.num_reg_images())
    return best
