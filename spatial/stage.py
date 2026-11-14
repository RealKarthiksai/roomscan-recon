"""Stage bookkeeping: wall time, peak RAM (psutil sampling), versions, hashes -> meta.json."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import psutil

from formats.models import StageMeta


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def sha256_tree(paths: list[Path]) -> str:
    """Order-independent digest of many files (name + content hash)."""
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(sha256_file(p).encode())
    return h.hexdigest()


def versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for mod in ["numpy", "open3d", "pycolmap", "torch", "transformers", "cv2"]:
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 - optional modules
            pass
    try:
        out["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
    except OSError:
        pass
    return out


class _PeakRam:
    """Samples RSS of this process + children every `dt` seconds."""

    def __init__(self, dt: float = 0.05):
        self.dt, self.peak, self._stop = dt, 0, threading.Event()
        self._proc = psutil.Process()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss + sum(
                    c.memory_info().rss for c in self._proc.children(recursive=True)
                )
            except psutil.Error:
                rss = 0
            self.peak = max(self.peak, rss)
            time.sleep(self.dt)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


class StageRecorder:
    """Collects metrics/outputs during a stage; `record_stage` writes meta.json at the end."""

    def __init__(self, stage: str, out_dir: Path, device: str, params: dict):
        self.stage, self.out_dir, self.device, self.params = stage, out_dir, device, params
        self.inputs: dict[str, str] = {}
        self.outputs: dict[str, str] = {}
        self.metrics: dict = {}
        self.notes: list[str] = []
        self.status = "ok"

    def add_input(self, path: Path, digest: str | None = None):
        self.inputs[str(path)] = digest or sha256_file(path)

    def add_output(self, path: Path):
        self.outputs[str(path.relative_to(self.out_dir.parent))] = sha256_file(path)


@contextmanager
def record_stage(stage: str, out_dir: Path, device: str = "cpu", params: dict | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = StageRecorder(stage, out_dir, device, params or {})
    started, t0 = datetime.now(UTC), time.perf_counter()
    vram = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        torch = None
    with _PeakRam() as ram:
        try:
            yield rec
        except BaseException as e:
            rec.status = "failed"
            rec.notes.append(f"{type(e).__name__}: {e}")
            raise
        finally:
            if torch is not None and torch.cuda.is_available():
                vram = torch.cuda.max_memory_allocated() / 2**20
            meta = StageMeta(
                stage=stage,
                status=rec.status,
                started=started.isoformat(),
                ended=datetime.now(UTC).isoformat(),
                wall_time_s=time.perf_counter() - t0,
                device=device,
                peak_ram_mb=ram.peak / 2**20,
                peak_vram_mb=vram,
                versions=versions(),
                params=rec.params,
                inputs=rec.inputs,
                outputs=rec.outputs,
                metrics=rec.metrics,
                notes=rec.notes,
            )
            (out_dir / "meta.json").write_text(meta.model_dump_json(indent=1))
