# Interfaces: roomscan-recon (Part 3 `spatial/`, Part 4 `recon/`)

Single source of truth for on-disk formats between phases. Pydantic models live in
`formats/`; every JSON file below validates against them. Version bumps follow semver
(patch = docs, minor = additive optional field, major = breaking).

## 0. Conventions (all phases)

| Item | Convention |
|---|---|
| Units | metres, radians; depth PNGs are uint16 **millimetres** (0 = invalid) |
| Time | int64 nanoseconds (`timestamp_ns`) |
| Camera frame | OpenCV: x right, y down, z forward |
| World frame | right-handed, **gravity-aligned, +Z up** (after Phase 2), metres |
| Pose | `T_world_camera` (camera→world), 4×4 row-major in JSON; COLMAP files store the inverse (`cam_from_world`), as COLMAP does |
| Intrinsics | pinhole `fx, fy, cx, cy` in pixels **for the stored image size**; images in a workspace are undistorted |
| Frame id | zero-padded string, `"000123"`, identical across `images/`, `depth/`, JSON keys |
| Ground truth | lives in `workspace/gt/` and is read **only** by `eval/`. Pipeline code must never read it |

## 1. Phase 1: ingest → workspace

`python -m spatial.cli ingest --dataset {replica,tum,eth3d} --src <path> --out runs/<run_id> [--max-frames N] [--stride K] [--downscale D] [--prior {gt,gt_noisy,none}] [--noise-preset {vio_small,vio_large}] [--seed S]`

```
runs/<run_id>/workspace/
  images/<frame>.jpg            undistorted RGB (quality 95) at stored resolution
  depth/<frame>.png             optional sensor depth, uint16 mm, registered to images
  depth_conf/<frame>.png        optional uint8 confidence (0 low … 255 high)
  priors.json                   formats.Priors (below)
  workspace_meta.json           formats.WorkspaceMeta
  gt/poses_gt.json              eval only: {frame: T_world_camera} in the dataset GT frame
  gt/gt_ref.json                eval only: path + sha256 of GT mesh/cloud, GT frame notes
runs/<run_id>/phase1/meta.json  formats.StageMeta
```

`priors.json` (`formats.Priors`, v0.1.0)

| Field | Type / unit | Meaning |
|---|---|---|
| `version` | semver | format version |
| `width`, `height` | px | stored image size |
| `intrinsics` | `{fx,fy,cx,cy}` px | shared pinhole intrinsics (after undistortion + downscale) |
| `frames[]` | list | per frame: `id`, `timestamp_ns`, `image`, `depth` (nullable), `depth_conf` (nullable) |
| `frames[].T_world_camera` | 4×4 or null | **prior** pose (VIO-like); null when `pose_source = none` |
| `pose_source` | enum `arkit, arcore, dataset_gt, dataset_gt_noisy, none` | where priors come from. `dataset_gt_noisy` = GT + simulated VIO drift (benchmark only) |
| `pose_noise` | object or null | noise preset + seed when `dataset_gt_noisy` |
| `gravity_world` | unit 3-vector or null | gravity direction (pointing **down**) in the prior world frame |
| `gravity_source` | enum `arkit, imu_accel, dataset_axis, none` | how gravity was obtained |
| `depth_scale_m` | float | metres per depth unit (0.001) |
| `depth_source` | enum `lidar, kinect, rendered, none` | sensor type of `depth/` |

## 2. Phase 2: pose refinement → refined workspace

`python -m spatial.cli refine --run runs/<run_id> [--max-image-size 1024] [--matcher {exhaustive,sequential}] [--device cpu|cuda]`

Adds to the workspace (inputs above are unchanged):
```
workspace/sparse/0/{cameras.bin, images.bin, points3D.bin}   metric, gravity-aligned (+Z up), our world frame
workspace/sparse/raw/                                        unaligned SfM model (kept as intermediate)
workspace/poses_refined.json                                 {frame: T_world_camera} (same frame as sparse/0)
workspace/alignment.json                                     Sim(3) raw→prior, gravity rotation, residuals
runs/<run_id>/phase2/meta.json
```
Rules: scale comes from the metric priors (Sim(3) RANSAC over camera centres); gravity
alignment rotates `gravity_world` to −Z. Frames not registered by SfM are listed in
`alignment.json.unregistered` and are absent from `sparse/0` and `poses_refined.json`.

## 3. Phase 3: depth + fusion

`python -m spatial.cli phase3 --workspace runs/<run_id>/workspace --depth-source {sensor,mono,auto} --model {small,base,large} --voxel 0.02 --device {cpu,cuda} --downscale N --max-frames N [--no-align] [--no-masks] [--no-consistency] [--pose-source {refined,gt}]`

Input: the Phase 2 workspace (`images/`, optional `depth/`, `depth_conf/`, `sparse/0/`, `priors.json`, `workspace_meta.json`). `--pose-source gt` exists only for the R0 oracle.

```
runs/<run_id>/phase3/
  depth_raw/<frame>.npy         cached model prediction (float32 m or relative), keyed by input+model hash
  depth_aligned/<frame>.png     uint16 mm, metric
  depth_valid/<frame>.png       uint8 {0,255}, mask after all enabled filters
  depth_alignment.json          formats.DepthAlignment
  fusion/cloud.ply              fused cloud: xyz (m, world), rgb, normals
  fusion/tsdf_mesh.ply          TSDF mesh, debugging/eval only (NOT a Part 4 output)
  meta.json                     formats.StageMeta
```

`depth_alignment.json` (`formats.DepthAlignment`)

| Field | Meaning |
|---|---|
| `model` | `{name, hf_id, revision, kind: metric|relative|inverse_relative, licence}` or `sensor` |
| `global_scale` | float or null (when global-scale option used) |
| `frames.<id>.source` | `sensor`, `mono` |
| `frames.<id>.scale`, `.shift` | depth = scale·pred + shift (shift 0 for metric models) |
| `frames.<id>.scale_smoothed` | after temporal median filter |
| `frames.<id>.inliers`, `.n_points` | RANSAC inliers / projected sparse points |
| `frames.<id>.residual_m` | median abs residual of inliers (m) |
| `frames.<id>.status` | `ok`, `rejected_few_points`, `rejected_residual`, `no_alignment` |

## 4. Phase 5 (HANDOFF P5): planes + room shell

`python -m spatial.cli room --cloud <run>/phase3/fusion/cloud.ply --out <run>/phase5 [--workspace <ws>] [--voxel 0.03] [--seed 0]`

Input: a fused cloud in the gravity-aligned world frame (+Z up, metres). With `--workspace`, the
command refuses a workspace whose `alignment.json` says `gravity_aligned: false`, and records the
scale source from `priors.json`. Code: `spatial/room/` (planes.py algorithm, run.py stage).

```
<run>/phase5/
  planes.json     formats.PlanesFile: planes[{id, label floor|ceiling|wall|other, normal (unit,
                  pointing into the room), d (n·x + d = 0), inlier_count, area_m2 (convex hull of
                  projected inliers), rms_m, polygon (hull, 3D)}], n_points, n_assigned, params
  room.json       formats.RoomModel: floor_polygon (2D xy, CCW), floor_polygon_source
                  (wall_intersections | floor_hull | none), floor_area_m2, floor_z/ceiling_z (median
                  inlier z), ceiling_height_m (perpendicular floor→ceiling distance), floor_id,
                  ceiling_id, wall_ids, bbox_min/max, gravity_world, scale_source,
                  scale_confidence (null: no estimator yet), notes[]
  meta.json       formats.StageMeta (metrics include floor_tilt_deg = residual frame tilt)
```

Rules: floor = lowest large horizontal plane facing up; ceiling = highest facing down; wall =
vertical plane with area ≥ 0.8 m², vertical span ≥ 50 % of room height, reaching within 0.35 m of
the ceiling (rejects cabinets and tables). Floor polygon = intersections of consecutive
*dominant* walls (largest wall per direction within 20°), validated against the floor hull;
recesses and bays are not modelled in v1.

## 5. `meta.json` for every stage (`formats.StageMeta`)

`stage, status (ok|failed|partial), started/ended (ISO 8601), wall_time_s, device, peak_ram_mb,
peak_vram_mb|null, versions{python, numpy, open3d, pycolmap, torch, transformers, git_commit},
params{…}, inputs{path: sha256}, outputs{path: sha256}, metrics{…}, notes[]`

## 6. Module boundaries

- `spatial/` (Part 3) and `recon/` (Part 4) never import each other.
- Both may import `formats/`. Only tests/benchmarks import `eval/`; pipeline code never does.
- `eval/` may read `workspace/gt/`; nothing else may.
