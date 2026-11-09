# insv-odometry

Extract IMU odometry and camera calibration from Insta360 `.insv` files, using
[telemetry-parser](https://github.com/AdrianEddy/telemetry-parser) (Python wheel `telemetry-parser`).

Tested with an **Insta360 X4 Air** clip (`VID_20260915_112738_00_002.insv`, 15.3 s, two 3840×3840
HEVC fisheye tracks @ 29.97 fps, 1 kHz IMU).

## Usage

```bash
uv sync
uv run insv-odometry extract path/to/VID_XXXXXXXX_XXXXXX_00_XXX.insv
uv run pytest
```

Options: `-o OUT` (default `out/<stem>`), `--kp/--ki` (Mahony gains), `--no-bias`, `--no-plot`.

## Outputs (`out/<stem>/`)

| file | content |
|---|---|
| `imu.csv` | raw IMU, 1 kHz: `t_s`, gyro (rad/s), accel (m/s², specific force, reads +g up at rest) |
| `orientation_imu.csv` | orientation at IMU rate: quaternion `qw qx qy qz` (IMU→world) + roll/pitch/yaw |
| `frames.csv` | one row per video frame: frame index, `t_s`, orientation (slerped), exposure time |
| `trajectory_tum.txt` | per-frame poses in TUM format (translation = 0, see below) |
| `calibration.json` | lens calibration (Mei/omni + RadtanPro per lens), rig extrinsics between the two lenses, readout time, IMU info |
| `metadata.json` | full Insta360 metadata (serial, firmware, offsets, gyro calib, …) |
| `summary.json` | sanity numbers (rates, coverage, total rotation, gravity correction size) |
| `odometry.png` | gyro, accel, orientation and gravity-correction plots |

Time: `t_s = 0` is the first video frame (telemetry-parser aligns IMU to video).
World frame: Z up; yaw = 0 at the start of the clip.

## What "odometry" means here

- **Orientation (3 DoF) is good.** Gyro integration with gravity correction (Mahony filter). On the
  test clip: 723° total rotation, and gyro-only vs gravity-corrected disagree by at most 6.4° (1.6°
  at the end). Roll and pitch are anchored to gravity. Yaw has no absolute reference (no
  magnetometer) and drifts slowly.
- **Translation is not extracted.** Double-integrating a consumer IMU accelerometer drifts by
  metres within seconds, and the camera has no GPS fix in this clip (`gps` metadata is all zeros).
  Translation must come from the video (phase 2), with the IMU giving rotation priors and metric
  scale.

Phase 2 checks these against visual rotation:
- **Time offset:** the IMU↔video offset is −2 ms.
- **IMU frame:** the normalized IMU frame is an axis permutation of the lens-0 camera frame.
  `fusion.json` holds the estimated rotation, accurate to 0.12° per step.

## Phase 2: 3D reconstruction fused with IMU

```bash
uv sync --extra recon
uv run --extra recon insv-odometry recon path/to/VID_XXXXXXXX_XXXXXX_00_XXX.insv
```

Options:
- `--step N`: use every Nth video frame (default 6, which is 5 fps)
- `--size`: virtual view size in px (default 768)
- `--max-features`, `--overlap`: SfM feature count and sequential matching overlap
- `--refine-rig`: let bundle adjustment refine the lens-to-lens extrinsics
- `--stages render,sfm,fuse,viewer`: run only some stages; later stages reuse the work folder
- `--video-width`: width of the viewer's 360 video (default 1920)

The pipeline is CPU-only (pycolmap wheel, no CUDA). Work goes to `out/<stem>/recon/`.

1. **render.** Both fisheye tracks are decoded and reprojected into **10 virtual pinhole views**
   (5 per lens: centre + 4 tilted 50°, 90° FOV) using the factory lens model.
   - Lens model: Mei unified model, `xi` + radtan from `offset_v3`, rescaled for the window crop.
   - Each view keeps its lens's optical centre, so the rig is exact, including the 3.35 cm offset
     between lenses.
   - Masks give every scene direction to exactly one view.
   - `equirect_*.jpg` stitches both lenses for a visual check of the lens model and extrinsics.
2. **sfm.** pycolmap rig SfM:
   - SIFT features, sequential matching with rig verification
   - `filter_stationary_matches` drops features that move with the camera (the operator)
   - incremental mapping with fixed intrinsics and rig
3. **fuse.** IMU fusion (`fusion.json`, `fusion.png`, `cloud.png`):
   - IMU↔video **time offset**: minimise per-step rotation-angle mismatch, SfM vs gyro
   - **camera→IMU rotation**: Kabsch on per-step rotation vectors
   - **gravity alignment**: Z up, from the gravity-corrected IMU orientation
   - **metric scale + accel bias**: second differences of SfM positions equal the
     triangle-weighted mean of IMU world acceleration, which gives an exact linear fit

4. **viewer.** A side-by-side web viewer in `out/<stem>/recon/viewer/`:
   - **Left:** the 360 video, levelled and world-locked using the fused poses (horizon flat,
     longitude 0 = overall walking direction).
   - **Right:** a three.js view of the metric sparse cloud, the camera path, and the moving
     front/back lens frustums, synced frame-by-frame to playback.
   - Poses between SfM keyframes come from the gyro, blended between neighbouring keyframes,
     so the 3D camera moves at the full 30 fps.
   - View modes: *Follow* chases the camera; dragging switches to free orbit; *Top* is a plan view.

   ```bash
   uv run insv-odometry serve            # newest out/*/recon/viewer -> http://127.0.0.1:8770/
   ```

   `serve` supports HTTP Range requests. Python's plain `http.server` does not, so the video
   could not seek.

Outputs: `trajectory_metric_tum.txt` (lens-0 camera, metres, Z up), `sparse_points_metric.ply`,
`sparse_best/` (COLMAP model), `fusion.json`, `viewer/`.

### Results on the test clip (X4 Air, 15.3 s, walking with the camera hand-held)

| check | result |
|---|---|
| registration | 77/77 frames, 770/770 views, 47k points, 0.86 px mean reprojection |
| IMU↔video time offset | −2.0 ms (parser timestamps already aligned) |
| rotation agreement after hand-eye | 0.12° RMS per step, vs 8.6° mean rotation per step |
| camera→IMU rotation | ≈180° about (x+z)/√2, an axis permutation |
| metric scale from IMU | **1.008 ± 0.018** (SfM was already metric via the fixed 3.35 cm lens offset; two independent sources agree within 1%) |
| accel fit | 0.22 m/s² RMS residual on 0.91 m/s² RMS signal |
| ground plane (gravity check) | 0.88° tilt, 2.6 cm RMS flatness; camera 1.24 m above ground (0.97–1.45 m) |
| path | 10.0 m walked, 2.5 × 5.4 m extent |
| viewer reprojection | sparse points projected into the levelled video land on the matching structures (trees, railing, pedestal, ground) |

Runtime on an i5-12500 (12 threads): render 42 s, features 103 s, matching 21 min, mapping
4 min, fusion seconds, viewer video 4 min. Matching dominates. Ways to cut it: `--overlap 4`, `--max-features 1500`,
or a larger `--step`.

### Next

- **Dense** (CPU): OpenMVS densify and mesh on the posed views. COLMAP PatchMatch MVS and Gaussian
  splatting need CUDA, so they are deferred to a GPU host.
- Rolling-shutter correction from the gyro (15.9 ms readout). The pipeline currently ignores it.
- Tighter gravity alignment: per-frame alignment spread is 1.9° median, coming from Mahony tilt
  error during walking.

### Dive lens / underwater footage

The camera is also used with the Insta360 dive lens. Behind a port in water, refraction changes
the effective focal length and distortion, so the factory air calibration in the file does not
hold.
- Phase 1 orientation does not depend on lens calibration and is unaffected.
- The `offset_states` metadata field shows the lens mode. The test clip reads `Common`, meaning
  normal mode, shot in air.
- For dive footage: start from the factory model, let bundle adjustment refine focal length and
  distortion per lens (self-calibration), and keep the rig rotation fixed.
- If self-calibration is unstable (low texture, turbid water), calibrate once with a checkerboard
  filmed underwater through the dive lens and use that model instead.
