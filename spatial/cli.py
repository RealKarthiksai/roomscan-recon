"""`python -m spatial.cli <command>`: Part 3 command-line entry points."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, help="Room Scanner Part 3: spatial understanding.")


def _run_dir(out: Path | None, tag: str) -> Path:
    return out or Path("runs") / f"{datetime.now():%Y%m%d-%H%M%S}_{tag}"


@app.command()
def ingest(
    dataset: Annotated[str, typer.Option(help="replica | tum | eth3d")],
    src: Annotated[Path, typer.Option(help="dataset scene directory")],
    out: Annotated[Path | None, typer.Option(help="run directory")] = None,
    max_frames: int | None = None,
    stride: int = 1,
    downscale: int = 1,
    prior: Annotated[str, typer.Option(help="gt | gt_noisy | none")] = "gt_noisy",
    noise_preset: str = "vio_small",
    seed: int = 0,
):
    """Phase 1: dataset -> workspace."""
    from spatial.io.ingest import ingest as _ingest

    run = _run_dir(out, f"{dataset}")
    ws = _ingest(
        dataset,
        src,
        run,
        max_frames=max_frames,
        stride=stride,
        downscale=downscale,
        prior=prior,
        noise_preset=noise_preset,
        seed=seed,
    )
    typer.echo(json.dumps({"workspace": str(ws)}))


@app.command()
def refine(
    run: Annotated[Path, typer.Option(help="run directory containing workspace/")],
    max_image_size: int = 1024,
    matcher: Annotated[str, typer.Option(help="auto | exhaustive | seq")] = "auto",
    prior_mode: Annotated[str, typer.Option(help="sim3 | ba")] = "sim3",
):
    """Phase 2: pose refinement -> metric, gravity-aligned sparse/0."""
    from spatial.sfm.refine import refine as _refine

    r = _refine(
        run / "workspace",
        run,
        max_image_size=max_image_size,
        matcher=matcher,
        prior_mode=prior_mode,
    )
    typer.echo(
        json.dumps({k: v for k, v in r.items() if not k.startswith(("sim3_R", "R_"))}, default=str)
    )


@app.command()
def phase3(
    workspace: Annotated[Path, typer.Option(help="Phase 2 workspace directory")],
    out: Annotated[Path | None, typer.Option(help="output dir (default <run>/phase3)")] = None,
    depth_source: Annotated[str, typer.Option(help="sensor | mono | auto")] = "auto",
    model: Annotated[
        str, typer.Option(help="small | base | large (base/large: CC-BY-NC)")
    ] = "small",
    model_kind: Annotated[str, typer.Option(help="metric | relative")] = "metric",
    voxel: float = 0.02,
    device: Annotated[str, typer.Option(help="cpu | cuda")] = "cpu",
    downscale: int = 1,
    max_frames: int | None = None,
    no_align: bool = False,
    no_masks: bool = False,
    no_consistency: bool = False,
    global_scale: bool = False,
    pose_source: Annotated[str, typer.Option(help="refined | gt (oracle only)")] = "refined",
):
    """Phase 3: depth maps + fused point cloud (docs/INTERFACES.md §3)."""
    from dataclasses import replace

    from spatial.fusion.tsdf import TSDFParams
    from spatial.phase3 import Phase3Config, run_phase3

    cuda = device.startswith("cuda")
    cfg = Phase3Config(
        depth_source=depth_source,
        model=model,
        model_kind=model_kind,
        device=device,
        downscale=downscale,
        max_frames=max_frames,
        align=not no_align,
        masks=not no_masks,
        consistency=not no_consistency,
        global_scale=global_scale,
        pose_source=pose_source,
        tsdf_p=TSDFParams(
            voxel=voxel,
            backend="tensor" if cuda else "legacy",
            device="CUDA:0" if cuda else "CPU:0",
        ),
    )
    cfg = replace(cfg)
    m = run_phase3(workspace, out or workspace.parent / "phase3", cfg)
    typer.echo(
        json.dumps(
            {
                k: m[k]
                for k in (
                    "depth_source",
                    "n_frames",
                    "n_points",
                    "frames_ok",
                    "frames_rejected",
                    "timings",
                )
            },
            default=str,
        )
    )


@app.command()
def room(
    cloud: Annotated[Path, typer.Option(help="Phase 3 fused cloud (fusion/cloud.ply)")],
    out: Annotated[Path, typer.Option(help="output dir (e.g. <run>/phase5)")],
    workspace: Annotated[
        Path | None, typer.Option(help="workspace, to record the scale source from priors.json")
    ] = None,
    voxel: float = 0.03,
    seed: int = 0,
):
    """Phase 5: planes + room shell -> planes.json, room.json (docs/INTERFACES.md §4)."""
    from formats.models import Priors
    from spatial.room.planes import PlaneParams
    from spatial.room.run import run_room

    scale_source = "unknown"
    if workspace is not None:
        al = workspace / "alignment.json"
        if al.exists() and not json.loads(al.read_text()).get("gravity_aligned", True):
            raise typer.BadParameter(
                f"{al}: world frame is not gravity-aligned (no gravity prior); floor/ceiling/"
                "wall labels would be meaningless"
            )
        pri = Priors.model_validate_json((workspace / "priors.json").read_text())
        if pri.pose_source in ("arkit", "arcore", "dataset_gt", "dataset_gt_noisy"):
            scale_source = "metric_priors"
        elif pri.depth_source in ("lidar", "kinect"):
            scale_source = "sensor_depth"
    r = run_room(cloud, out, PlaneParams(voxel=voxel, seed=seed), scale_source)
    typer.echo(json.dumps(r["metrics"], default=str))


if __name__ == "__main__":
    app()
