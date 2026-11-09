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

Not yet verified:
- the exact IMU→lens rotation (rig says `imu_from_rig` = identity, Insta360 also stores
  `imu_turn_ypr` = [0.1, 0.2, 90.5]°)
- the IMU↔video time offset

Phase 2 checks both against visual rotation.

## Phase 2: 3D reconstruction with IMU (plan)

1. **Frames → virtual pinhole views.** Decode both fisheye tracks with PyAV. Reproject each lens
   with its Mei calibration into 4–6 pinhole views (≈90° FOV each). All views share the known lens
   extrinsics, so each time step is one multi-camera rig.
2. **SfM with rig constraints.** Run COLMAP/GLOMAP via `pycolmap` (rig + frame support), sequential
   matching (ALIKED/LightGlue or SIFT). Subsample to ~5–10 fps. 458 frames at 30 fps is far more
   than SfM needs.
3. **Fuse IMU.**
   - Check sync and extrinsics: cross-correlate SfM angular velocity with the gyro to get the time
     offset, then solve hand-eye for IMU→rig rotation.
   - Gravity-align the reconstruction (Z up).
   - Recover **metric scale** by fitting SfM camera accelerations (spline second derivative) to IMU
     specific force minus gravity.
   - Optional: IMU rotation priors in bundle adjustment for weak-texture segments.
4. **Dense.** CPU-only first: the current host has an i5-12500, 16 GB RAM and no CUDA GPU.
   - Use OpenMVS (CPU) densify and mesh on the posed pinhole views, at reduced resolution.
   - COLMAP PatchMatch MVS and Gaussian splatting need CUDA, so they are deferred to a GPU host.
5. **Outputs.** Metric-scaled trajectory (TUM), sparse + dense point cloud (`.ply`), and a
   comparison plot of SfM rotation vs IMU rotation.

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
