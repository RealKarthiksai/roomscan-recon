"""Phase 5 stage: fused cloud -> planes.json + room.json (+ meta.json). docs/INTERFACES.md §4."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import open3d as o3d

from formats.models import Plane, PlanesFile, RoomModel
from spatial.room.planes import PlaneParams, detect_planes, floor_polygon, label_planes
from spatial.stage import record_stage


def run_room(
    cloud: Path,
    out: Path,
    p: PlaneParams | None = None,
    scale_source: str = "unknown",
) -> dict:
    """`cloud`: fused cloud in the gravity-aligned world frame (+Z up, metres)."""
    p = p or PlaneParams()
    with record_stage("phase5_room", out, device="cpu", params=asdict(p)) as rec:
        rec.add_input(cloud)
        pcd = o3d.io.read_point_cloud(str(cloud))
        if len(pcd.points) == 0:
            raise ValueError(f"empty or unreadable cloud: {cloud}")
        P, planes = detect_planes(pcd, p)
        lab = label_planes(P, planes, p)
        poly, area, src = floor_polygon(P, planes, lab, p)

        out_planes = []
        for k, pl in enumerate(planes):
            polygon, a = pl.hull(P)
            out_planes.append(
                Plane(
                    id=k,
                    label=pl.label,
                    normal=pl.normal.tolist(),
                    d=pl.d,
                    inlier_count=len(pl.idx),
                    area_m2=a,
                    rms_m=pl.rms(P),
                    polygon=polygon,
                )
            )
        n_assigned = int(sum(len(pl.idx) for pl in planes))
        pf = PlanesFile(planes=out_planes, n_points=len(P), n_assigned=n_assigned, params=asdict(p))
        notes = []
        if lab["floor"] is None:
            notes.append("no floor plane found")
        if lab["ceiling"] is None:
            notes.append("no ceiling plane found (ceiling unseen or not planar)")
        if src != "wall_intersections":
            notes.append(f"floor polygon from {src}: wall loop not closed or failed validation")
        fz, cz = lab["floor_z"], lab["ceiling_z"]
        room = RoomModel(
            floor_polygon=poly.tolist(),
            floor_polygon_source=src,
            floor_area_m2=area,
            floor_z=fz,
            ceiling_z=cz,
            ceiling_height_m=lab["ceiling_height"],
            floor_id=lab["floor"],
            ceiling_id=lab["ceiling"],
            wall_ids=lab["walls"],
            bbox_min=P.min(0).tolist(),
            bbox_max=P.max(0).tolist(),
            gravity_world=[0.0, 0.0, -1.0],
            scale_source=scale_source,
            scale_confidence=None,
            notes=notes,
        )
        (out / "planes.json").write_text(pf.model_dump_json(indent=1))
        (out / "room.json").write_text(room.model_dump_json(indent=1))
        rec.add_output(out / "planes.json")
        rec.add_output(out / "room.json")
        rec.metrics |= {
            "n_planes": len(planes),
            "labels": {
                lbl: sum(q.label == lbl for q in planes)
                for lbl in ("floor", "ceiling", "wall", "other")
            },
            "assigned_fraction": n_assigned / max(1, len(P)),
            "floor_polygon_source": src,
            "floor_area_m2": area,
            "ceiling_height_m": room.ceiling_height_m,
            "n_corners": len(poly),
            "floor_tilt_deg": (
                float(np.degrees(np.arccos(min(1.0, abs(planes[lab["floor"]].normal[2])))))
                if lab["floor"] is not None
                else None
            ),
        }
        rec.notes += notes
        return {"planes": planes, "P": P, "room": room, "metrics": rec.metrics}


def wall_floor_angles_deg(planes: list[Plane]) -> list[float]:
    """|90° - angle(floor normal, wall normal)| per wall (0 = perfectly perpendicular)."""
    fl = [q for q in planes if q.label == "floor"]
    if not fl:
        return []
    nf = np.asarray(fl[0].normal)
    return [
        float(abs(90.0 - np.degrees(np.arccos(np.clip(abs(nf @ np.asarray(w.normal)), 0, 1)))))
        for w in planes
        if w.label == "wall"
    ]
