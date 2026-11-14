"""Phase 5: planes + room (HANDOFF §5 P5). Measured values -> runs/test_results/phase5_*.json.

light: synthetic room with exact GT (eval/synthetic_room.py), schema + determinism.
heavy: Replica room0 Phase 3 sensor-depth cloud (runs/bench_phase3/room0/R1): wall ⟂ floor,
       and ceiling height / floor area vs the same detector run on the GT mesh.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from eval.synthetic_room import fused_cloud, gt
from formats.models import PlanesFile, RoomModel
from spatial.room.run import run_room, wall_floor_angles_deg
from spatial.stage import sha256_file

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "runs" / "test_results"


def _save(name, values):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"phase5_{name}.json").write_text(json.dumps(values, indent=1, default=str))


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    d = tmp_path_factory.mktemp("p5_syn")
    o3d.io.write_point_cloud(str(d / "cloud.ply"), fused_cloud(seed=0))
    r = run_room(d / "cloud.ply", d / "phase5", scale_source="synthetic")
    return d, r


def _match(gtp, planes, label):
    """Detected plane with `label`, normal within 10° of GT, nearest to the GT plane point."""
    n_gt, c = np.asarray(gtp["normal"]), np.asarray(gtp["point"])
    cands = [q for q in planes if q.label == label and q.normal @ n_gt > np.cos(np.radians(10))]
    if not cands:
        return None
    q = min(cands, key=lambda q: abs(q.normal @ c + q.d))
    return {
        "normal_err_deg": float(np.degrees(np.arccos(np.clip(q.normal @ n_gt, -1, 1)))),
        "offset_err_m": float(abs(q.normal @ c + q.d)),
    }


@pytest.mark.light
def test_synthetic_room_matches_gt(synthetic):
    d, r = synthetic
    G, room = gt(), r["room"]
    per = {
        k: _match(v, r["planes"], k.split("_")[0]) for k, v in G["planes"].items()
    }  # floor, ceiling, wall_*
    poly = np.asarray(room.floor_polygon)
    corner_err = [
        float(np.min(np.linalg.norm(poly - np.asarray(c), axis=1))) for c in G["corners_xy"]
    ]
    vals = {
        "planes": per,
        "ceiling_height_m": room.ceiling_height_m,
        "ceiling_height_err_m": abs(room.ceiling_height_m - G["ceiling_height_m"]),
        "floor_area_m2": room.floor_area_m2,
        "floor_area_rel_err": abs(room.floor_area_m2 - G["floor_area_m2"]) / G["floor_area_m2"],
        "floor_polygon_source": room.floor_polygon_source,
        "corner_err_m": corner_err,
        "labels": r["metrics"]["labels"],
        "wall_time_s": json.loads((d / "phase5" / "meta.json").read_text())["wall_time_s"],
    }
    _save("synthetic", vals)
    assert all(per.values()), f"GT plane not detected: {per}"
    assert max(v["normal_err_deg"] for v in per.values()) <= 2.0
    assert max(v["offset_err_m"] for v in per.values()) <= 0.02
    assert vals["ceiling_height_err_m"] <= 0.02
    assert vals["floor_area_rel_err"] <= 0.03
    assert room.floor_polygon_source == "wall_intersections" and len(poly) == 4
    assert r["metrics"]["labels"]["wall"] == 4  # cabinet/table faces are not walls


@pytest.mark.light
def test_outputs_schema_valid_and_deterministic(synthetic, tmp_path):
    d, _ = synthetic
    PlanesFile.model_validate_json((d / "phase5" / "planes.json").read_text())
    RoomModel.model_validate_json((d / "phase5" / "room.json").read_text())
    run_room(d / "cloud.ply", tmp_path / "again", scale_source="synthetic")
    for f in ("planes.json", "room.json"):
        assert sha256_file(d / "phase5" / f) == sha256_file(tmp_path / "again" / f), f


REPLICA_CLOUD = ROOT / "runs/bench_phase3/room0/R1/phase3/fusion/cloud.ply"
REPLICA_MESH = ROOT / "data/replica/room0_mesh.ply"


@pytest.mark.heavy
def test_replica_room0_real_cloud(tmp_path):
    if not (REPLICA_CLOUD.exists() and REPLICA_MESH.exists()):
        pytest.skip("needs Phase 3 bench run R1 (benchmarks.phase3) and data/replica")
    from eval.gt import load_mesh
    from eval.recon_metrics import sample_mesh

    r = run_room(REPLICA_CLOUD, tmp_path / "pred", scale_source="metric_priors")
    g = sample_mesh(load_mesh(REPLICA_MESH), 400000, seed=0)
    g.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(0.09, 30))
    o3d.io.write_point_cloud(str(tmp_path / "gt.ply"), g)
    rg = run_room(tmp_path / "gt.ply", tmp_path / "gt", scale_source="synthetic")
    pf = PlanesFile.model_validate_json((tmp_path / "pred" / "planes.json").read_text())
    dev = wall_floor_angles_deg(pf.planes)
    p, q = r["room"], rg["room"]
    vals = {
        "wall_floor_deviation_deg": dev,
        "n_walls": len(dev),
        "floor_tilt_deg": r["metrics"]["floor_tilt_deg"],
        "ceiling_height_m": p.ceiling_height_m,
        "ceiling_height_gtmesh_m": q.ceiling_height_m,
        "floor_area_m2": p.floor_area_m2,
        "floor_area_gtmesh_m2": q.floor_area_m2,
        "floor_polygon_source": [p.floor_polygon_source, q.floor_polygon_source],
        "wall_time_s": json.loads((tmp_path / "pred" / "meta.json").read_text())["wall_time_s"],
        "peak_ram_mb": json.loads((tmp_path / "pred" / "meta.json").read_text())["peak_ram_mb"],
    }
    _save("replica_room0", vals)
    assert len(dev) >= 4 and max(dev) <= 3.0  # HANDOFF P5 real-data check: floor ⟂ wall
    assert abs(p.ceiling_height_m - q.ceiling_height_m) <= 0.02
    assert abs(p.floor_area_m2 - q.floor_area_m2) / q.floor_area_m2 <= 0.03
