"""Phase 3 light tests: measured behaviour of alignment, masks, consistency, fusion parity."""

import numpy as np
import pytest

from spatial.depth.align import AlignParams, apply_alignment, fit_scale, smooth_scales
from spatial.depth.consistency import ConsistencyParams, consistency_mask, select_neighbours
from spatial.depth.masks import MaskParams, build_mask


@pytest.mark.light
@pytest.mark.parametrize("kind", ["metric", "inverse_relative"])
def test_alignment_recovers_scale_with_20pct_outliers(kind):
    rng = np.random.default_rng(0)
    z = rng.uniform(0.5, 5.0, 500)  # metric sparse depths
    s_true, t_true = 1.37, 0.08
    if kind == "metric":
        p = z / s_true
    else:  # 1/z = s*p + t
        p = (1 / z - t_true) / s_true
    p = p * rng.normal(1, 0.005, len(p))  # 0.5 % prediction noise
    out = rng.random(len(z)) < 0.20  # 20 % gross outliers
    z_obs = z.copy()
    z_obs[out] *= rng.uniform(1.5, 3.0, out.sum())
    r = fit_scale(z_obs, p, kind, AlignParams())
    assert r["status"] == "ok"
    assert abs(r["scale"] - s_true) / s_true < 0.01
    d = apply_alignment(p, kind, r["scale"], r["shift"])
    assert np.median(np.abs(d[~out] - z[~out]) / z[~out]) < 0.01


@pytest.mark.light
def test_alignment_rejects_frames_with_too_few_points():
    r = fit_scale(np.array([1.0, 2.0]), np.array([0.5, 1.0]), "metric", AlignParams())
    assert r["status"] == "rejected_few_points"


@pytest.mark.light
def test_temporal_smoothing_removes_scale_spike():
    s = [1.0, 1.01, 0.99, 1.6, 1.0, 1.02, None, 0.98]
    sm = smooth_scales(s, 5)
    assert abs(sm[3] - 1.0) < 0.03 and sm[6] is None


def _wall_scene(n=6, W=96, H=72, f=80.0):
    """Cameras translating sideways in front of a wall at z=2 (camera z forward)."""
    K = np.array([[f, 0, W / 2 - 0.5], [0, f, H / 2 - 0.5], [0, 0, 1]])
    T = []
    for k in range(n):
        M = np.eye(4)
        M[0, 3] = 0.05 * k
        T.append(M)
    depths = [np.full((H, W), 2.0, np.float32) for _ in T]
    return K, T, depths


@pytest.mark.light
def test_consistency_removes_injected_inconsistent_depth():
    K, T, depths = _wall_scene()
    bad = np.zeros_like(depths[2], bool)
    bad[20:40, 30:60] = True
    depths[2][bad] = 2.4  # 20 % wrong in one frame only
    valid = [d > 0 for d in depths]
    p = ConsistencyParams(k_neighbours=4, min_views=2, rel_thresh=0.03, min_baseline_m=0.01)
    neigh = select_neighbours(T, p)
    m, _ = consistency_mask(2, depths, valid, T, K, neigh[2], p)
    assert not m[bad].any()  # every injected pixel removed
    good_checked = ~bad
    good_checked[:, -12:] = False  # right border: may leave all neighbours
    assert m[good_checked].mean() > 0.95


@pytest.mark.light
def test_edge_mask_removes_flying_pixels_only_at_discontinuity():
    K = np.array([[80.0, 0, 47.5], [0, 80.0, 35.5], [0, 0, 1]])
    d = np.full((72, 96), 3.0, np.float32)
    d[:, :48] = 1.5  # step edge at column 48
    m, stats = build_mask(d, K, MaskParams(edge_dilate=1))
    assert not m[:, 46:50].any()  # band around the step removed
    assert m[:, :44].all() and m[:, 52:].all()  # flat regions kept
    assert stats["edge"] > 0


@pytest.mark.light
@pytest.mark.parametrize("tilt_deg", [30, 60, 80])
def test_grazing_mask_matches_analytic_incidence(tilt_deg):
    """Plane z = 2 + tan(a)·X: per-pixel incidence angle is analytic; the mask must equal
    (angle < 75 deg) except within 1 deg of the threshold."""
    K = np.array([[80.0, 0, 47.5], [0, 80.0, 35.5], [0, 0, 1]])
    H, W = 72, 96
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x, y = (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]
    a = np.radians(tilt_deg)
    d = (2.0 / (1 - np.tan(a) * x)).astype(np.float32)
    ok = (d > 0) & (d < 50)
    d[~ok] = 0
    m, _ = build_mask(d, K, MaskParams(max_m=50, edge_rel=1.0))
    n = np.array([-np.sin(a), 0, np.cos(a)])  # plane normal
    ray = np.stack([x, y, np.ones_like(x)], -1)
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
    ang = np.degrees(np.arccos(np.abs(ray @ n)))
    sure = ok & (np.abs(ang - 75) > 1)
    sure[[0, -1], :] = sure[:, [0, -1]] = False  # image border (one-sided diffs)
    assert (m[sure] == (ang[sure] < 75)).mean() > 0.999


@pytest.mark.light
def test_tsdf_backends_agree_on_cpu():
    """legacy ScalableTSDF vs tensor VoxelBlockGrid (CPU:0) on a ray-cast sphere."""
    import open3d as o3d

    from spatial.fusion.tsdf import TSDF, TSDFParams

    centre = np.array([0.0, 0.0, 2.0])
    mesh = o3d.geometry.TriangleMesh.create_sphere(0.5, 96).translate(centre)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    W, H, f = 160, 120, 128.0
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    vols = {b: TSDF(TSDFParams(voxel=0.01, backend=b)) for b in ("legacy", "tensor")}
    for ang in np.linspace(-np.pi, np.pi, 8, endpoint=False):
        c = centre + 2.0 * np.array([np.sin(ang), 0.0, -np.cos(ang)])
        z = (centre - c) / 2.0
        x = np.cross([0, -1, 0], z)
        x /= np.linalg.norm(x)
        T = np.eye(4)
        T[:3, :3] = np.c_[x, np.cross(z, x), z]
        T[:3, 3] = c
        T_cw = np.linalg.inv(T)
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            o3d.core.Tensor(K), o3d.core.Tensor(T_cw), W, H
        )
        t = scene.cast_rays(rays)["t_hit"].numpy()
        d = np.where(np.isfinite(t), t * (rays.numpy()[..., 3:] @ T_cw[:3, :3].T)[..., 2], 0)
        rgb = np.full((H, W, 3), 128, np.uint8)
        for v in vols.values():
            v.integrate(d.astype(np.float32), rgb, K, T)
    radii = {}
    for b, v in vols.items():
        pcd, _ = v.extract()
        r = np.linalg.norm(np.asarray(pcd.points) - centre, axis=1)
        assert len(r) > 1000, b
        radii[b] = (r.mean(), r.std())
    for b, (m, s) in radii.items():
        assert abs(m - 0.5) < 0.005 and s < 0.005, (b, m, s)
