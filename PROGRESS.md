# PROGRESS: roomscan-recon

Room Scanner Part 3 (`spatial/`) + Part 4 (`recon/`). This machine: Intel Core Ultra 200V,
15 GB RAM (~3 GB free in practice), Intel Arc iGPU (**no CUDA**), CPU-only runs.
GPU/ARKitScenes work is listed under "Pending for the GPU test machine".

| Phase | Scope | Status |
|---|---|---|
| 0 | Environment, pinned deps, Docker tools, env acceptance tests | done, except GPU tests (pending) |
| 1 | Dataset adapters → workspace (`images/ depth/ priors.json gt/`) | done (Replica, TUM, ETH3D) |
| 2 | Pose refinement → metric, gravity-aligned `sparse/0` | done; 1 documented known miss |
| 3 | Depth sources, alignment, masks, consistency, TSDF fusion + benchmarks | in progress |
| 5 | Planes + room shell (`spatial/room/`, HANDOFF P5) | done on synthetic + Replica; ETH3D check blocked (no gravity) |
| 6+ | SpatialMap writer (P6), Part 4 reconstruction | not started |

## Phase 0 evidence (2026-11-12, CPU)

| Test | Result |
|---|---|
| E1 versions | pass: torch 2.14.1+cpu, gsplat 1.5.3, open3d 0.19.0, pycolmap 4.2.1 |
| E2 RANSAC plane | pass: normal error < 0.01°, d = −0.5001, 5009 inliers |
| E3 ScalableTSDF sphere (8 views) | pass: radius 0.50019 m (GT 0.5), std 2.4 mm |
| E6 pycolmap SfM, ETH3D office 26 imgs @1600 px | **FAIL**: 20/26 registered (77 % < 90 %), reproj 1.083 px; 48 s. Likely low texture + downscale; re-test at full res on GPU machine |
| E4/E5 gsplat | pending (CUDA) |
| E7 OpenMVS image | pass: all 5 apps report `OpenMVS x64 v2.4.0`, no loader errors; image 594 MB |

## Phase 1 evidence (2026-11-13, CPU; `runs/test_results/phase1_*.json`)

| Check | Result |
|---|---|
| ETH3D pose convention: reproject GT points with converted poses | median 0.84 px, p95 3.05 px (11,687 obs) |
| Replica pose convention: ray-cast GT mesh vs rendered depth (10 frames) | median abs diff 0.038 mm (= depth quantisation) |
| Replica gravity: floor-plane tilt vs +Z | 0.26° → gravity = −Z confirmed |
| TUM rgb↔depth association | max Δt 17.2 ms (≤ 20 ms) |
| TUM gravity from accelerometer (motion-compensated) | spread 2.88° over 1468 samples; axis map diag(−1,−1,1); 4.29° from mocap Z |
| Simulated VIO prior (light) | yaw-only drift, grows with path |

## Phase 2 evidence (2026-11-13, CPU; `runs/test_results/phase2_*.json`)

| Dataset (frames, prior) | Registered | Reproj | ATE SE(3) prior → refined | Scale err | Rot err | Time / peak RAM |
|---|---|---|---|---|---|---|
| ETH3D office (26, none: SfM only, Sim(3) eval) | 26/26 | 0.47 px | median 0.42 cm (Sim3) | n/a | 0.16° | 17.6 s / 4.3 GB |
| TUM fr1/xyz (60, vio_large) | 60/60 | 0.89 px | 3.87 → **1.13 cm** (−71 %) | 0.10 % | 1.83°* | 47.6 s / 1.4 GB |
| Replica room0 (80, vio_small) | 80/80 | 0.46 px | 4.67 → **0.63 cm** (−87 %) | 0.39 % | 0.13° | 17.8 s / 1.2 GB |
| Replica room0 (80, vio_large) | 80/80 | 0.46 px | 17.57 → 2.63 cm (−85 %) **MISS > 2 cm** | 2.35 % | 0.14° | — |

\* TUM GT orientation carries a camera↔mocap offset: SfM rotations match ETH3D's laser-registered
GT to 0.16° and Replica's exact GT to 0.13°, so TUM is judged on translation only.

Known issues:
- **Replica vio_large miss**: the shape is right (Sim(3) ATE 0.49 cm), but the scale comes from priors
  drifting 3 %/m over a 23 m path. Kept as an xfail with the measured cause; the fix is depth-based scale (Phase 3).
- **ETH3D registration is intermittent**: 1 of ~9 runs registered only 20/26 (multithreaded COLMAP
  isn't bit-deterministic even with fixed seeds). Mitigated by seeding and `multiple_models=True` (keep the largest).
  Repeat runs: ATE 0.91–1.65 cm, rot 0.16–0.28°.

## Phase 5 evidence (2026-11-14, CPU; `runs/test_results/phase5_*.json`, `pytest tests/test_phase5.py`)

| Check | Result (target) |
|---|---|
| Synthetic room 4×3.5×2.6 m + table + cabinet, 40 ray-cast views, σ 3 mm, TSDF 2 cm | max normal err 0.23° (≤ 2°), max offset 4.3 mm (≤ 2 cm), ceiling height err 0.02 mm (≤ 2 cm), floor area err 0.34 % (≤ 3 %), max corner err 9.6 mm; 4 walls, cabinet/table rejected; 1.7 s |
| Determinism: rerun on same cloud | planes.json, room.json byte-identical |
| Replica room0 R1 cloud (sensor depth, refined poses) | 9 walls, wall⟂floor max 0.41° (≤ 3°); ceiling 2.7293 m vs 2.7279 m from GT mesh; floor area 35.38 vs 35.32 m² (0.17 %); 7.9 s, 0.8 GB peak |
| ETH3D office floor⟂wall | **not run**: SfM-only workspace is not gravity-aligned or metric; `room` CLI refuses it. Mono Phase 3 cloud there is also too sparse (0.28 % valid pixels) |

Known issues:
- Replica room0 refined frame is tilted 1.74° (floor normal vs +Z): a Phase 2 gravity residual. Ceiling height uses perpendicular plane distance, so it is unaffected.
- Floor polygon uses one dominant wall per direction; recesses/bays (room0 has two) are approximated. The GT-mesh comparison is self-consistency, not an independent layout GT.
- `scale_confidence` is null (no estimator yet).

## Decisions and findings log

- **Replica meshes have quad faces**; open3d 0.19's PLY reader silently keeps only triangles
  (16,516 of 953,647 faces). `eval/gt.py` loads with plyfile and splits quads.
- Phase 2 `prior_mode`: `sim3` (pure SfM, then robust Sim(3) to priors) is the default; `ba`
  (prior positions inside BA) gave identical accuracy on TUM (ATE 1.12 vs 1.14 cm) and is slower.
- Integer downscale uses the pixel-centre rule `c' = (c + 0.5)/D − 0.5` (light test).

- open3d **0.20.0 rejected**: `ScalableTSDFVolume` and tensor `VoxelBlockGrid` return 0 vertices
  (plane at 2 m, 8-view sphere); `UniformTSDFVolume` works. 0.19.0 works with numpy 2.5.4.
- nerfstudio 1.1.5 pins gsplat 1.4.0 + opencv-headless 4.10, so it's excluded; gsplat used directly.
- OpenMVS v2.4.0 needs OpenCV ≥ 4.10 → Ubuntu 26.04 base; two source patches (Boost.System
  CMake component, `IMWRITE_JPEGXL_QUALITY` constant). See `docker/openmvs/Dockerfile`.
- Depth-Anything-V2 **Small = Apache-2.0** (default); Base/Large = CC-BY-NC-4.0 (opt-in only).
- Phase 5 wall rule: vertical, ≥ 0.8 m², must reach within 0.35 m of the ceiling. Without the
  ceiling rule the synthetic cabinet front was labelled a wall and shifted a corner 0.4 m.
- Phase 5 horizontal-plane height = median inlier z (z at x=y=0 picked an 8.7°-tilted patch as floor on the Replica GT mesh).
- `tests/test_phase3_heavy.py` had two orphan lines (syntax error that broke collection); removed, no assertion changed.
- No `/usr/bin/time` on this machine: peak RAM is measured with psutil sampling.

## Pending for the GPU test machine

- E4/E5 gsplat tests; all `pytest -m gpu` tests (CUDA fusion, GPU inference parity).
- All `pytest -m arkit` tests (ARKitScenes: R0–R6 on real LiDAR, Step D on ARKitScenes).
- Re-run E6 at full resolution.
