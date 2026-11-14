"""Deterministic synthetic room with exact ground truth (HANDOFF §5, P5).

Box room 4.0 x 3.5 x 2.6 m (x, y, z; floor z=0, +Z up), a table (top at 0.76 m, four legs)
and a tall cabinet against the y=0 wall. Depth is ray-cast (Open3D RaycastingScene) from a
seeded camera set inside the room, with additive Gaussian noise, and fused with the Phase 3
TSDF (spatial.fusion.tsdf). Only tests/benchmarks import this module.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

ROOM = (4.0, 3.5, 2.6)


def _box(size, origin) -> o3d.geometry.TriangleMesh:
    m = o3d.geometry.TriangleMesh.create_box(*size)
    m.translate(origin)
    return m


def room_mesh() -> o3d.geometry.TriangleMesh:
    Lx, Ly, Lz = ROOM
    m = _box(ROOM, (0, 0, 0))
    m += _box((1.2, 0.8, 0.04), (1.0, 1.2, 0.72))  # table top: z in [0.72, 0.76]
    for x in (1.02, 2.13):
        for y in (1.22, 1.93):
            m += _box((0.05, 0.05, 0.72), (x, y, 0.0))
    m += _box((0.8, 0.4, 1.8), (2.9, 0.0, 0.0))  # cabinet against wall y=0
    return m


def gt() -> dict:
    """GT planes (inward normals, n.x + d = 0), room dimensions."""
    Lx, Ly, Lz = ROOM
    planes = {
        "floor": ([0, 0, 1.0], 0.0, [Lx / 2, Ly / 2, 0]),
        "ceiling": ([0, 0, -1.0], Lz, [Lx / 2, Ly / 2, Lz]),
        "wall_x0": ([1.0, 0, 0], 0.0, [0, Ly / 2, Lz / 2]),
        "wall_x1": ([-1.0, 0, 0], Lx, [Lx, Ly / 2, Lz / 2]),
        "wall_y0": ([0, 1.0, 0], 0.0, [Lx / 2, 0, Lz / 2]),
        "wall_y1": ([0, -1.0, 0], Ly, [Lx / 2, Ly, Lz / 2]),
    }
    return {
        "planes": {k: {"normal": n, "d": d, "point": c} for k, (n, d, c) in planes.items()},
        "ceiling_height_m": Lz,
        "floor_area_m2": Lx * Ly,
        "corners_xy": [[0, 0], [Lx, 0], [Lx, Ly], [0, Ly]],
    }


def look(position, yaw_deg, pitch_deg) -> np.ndarray:
    """T_world_camera, OpenCV camera (x right, y down, z forward), world +Z up."""
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    f = np.array([np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), np.sin(p)])
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    T = np.eye(4)
    T[:3, :3] = np.c_[r, np.cross(f, r), f]
    T[:3, 3] = position
    return T


def cameras(seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    c = np.array([ROOM[0] / 2, ROOM[1] / 2, 1.4])
    T = []
    for k, yaw in enumerate(np.arange(0, 360, 15.0)):
        pos = c + 0.6 * np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw)), 0])
        pos = pos + rng.normal(0, 0.05, 3)
        T.append(look(pos, yaw + 180 + rng.normal(0, 5), (-25 if k % 2 else 20) + rng.normal(0, 3)))
    for yaw in np.arange(0, 360, 45.0):
        T.append(look(c, yaw, -60.0))  # floor
        T.append(look(c, yaw + 22.5, 60.0))  # ceiling
    return T


def render_depth(scene, T_wc, K, W, H) -> np.ndarray:
    T_cw = np.linalg.inv(T_wc)
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        o3d.core.Tensor(K), o3d.core.Tensor(T_cw), W, H
    )
    t = scene.cast_rays(rays)["t_hit"].numpy()
    dirs_cam = rays.numpy()[..., 3:] @ T_cw[:3, :3].T
    return np.where(np.isfinite(t), t * dirs_cam[..., 2], 0).astype(np.float32)


def fused_cloud(
    seed: int = 0,
    noise_m: float = 0.003,
    W: int = 320,
    H: int = 240,
    f: float = 180.0,
    voxel: float = 0.02,
) -> o3d.geometry.PointCloud:
    from spatial.fusion.tsdf import TSDF, TSDFParams

    K = np.array([[f, 0, W / 2 - 0.5], [0, f, H / 2 - 0.5], [0, 0, 1.0]])
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(room_mesh()))
    rng = np.random.default_rng(seed + 1)
    vol = TSDF(TSDFParams(voxel=voxel))
    rgb = np.full((H, W, 3), 128, np.uint8)
    for T in cameras(seed):
        d = render_depth(scene, T, K, W, H)
        d = np.where(d > 0, d + rng.normal(0, noise_m, d.shape), 0).astype(np.float32)
        vol.integrate(d, rgb, K, T)
    pcd, _ = vol.extract()
    return pcd
