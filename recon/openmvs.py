"""OpenMVS (docker image `room-openmvs:2.4.0`, CPU) stage wrappers for Part 4.

All paths handed to the tools are relative to the mounted working folder `/out`, which is the
stage output directory, so scene files stay relocatable."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

IMAGE = "room-openmvs:2.4.0"
BIN = "/opt/openmvs/bin/OpenMVS"


class OpenMVSError(RuntimeError):
    pass


def available() -> bool:
    if shutil.which("docker") is None:
        return False
    r = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True)
    return r.returncode == 0


@dataclass
class ToolRun:
    tool: str
    args: list[str]
    seconds: float


def run_tool(
    tool: str, args: list[str], out: Path, images: Path | None = None, timeout: int = 7200
) -> ToolRun:
    cmd = [
        "docker",
        "run",
        "--rm",
        "-u",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{out.resolve()}:/out",
    ]
    if images is not None:
        cmd += ["-v", f"{images.resolve()}:/images:ro"]
    cmd += [IMAGE, f"{BIN}/{tool}", *args, "-w", "/out"]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    dt = time.perf_counter() - t0
    log = out / "logs"
    log.mkdir(exist_ok=True)
    (log / f"{tool}.log").write_text(
        "$ " + " ".join(cmd) + "\n" + r.stdout[-200_000:] + "\n" + r.stderr[-200_000:]
    )
    if r.returncode != 0:
        raise OpenMVSError(f"{tool} failed ({r.returncode}); see {log / (tool + '.log')}")
    return ToolRun(tool, args, dt)
