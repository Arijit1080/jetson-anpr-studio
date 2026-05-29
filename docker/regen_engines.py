"""Regenerate TensorRT `.engine` files from YOLO `.pt` weights.

Why this exists: TensorRT engines are NOT portable across TRT minor versions.
An engine built against TRT 10.3 won't load against TRT 10.5.  We can't bake
engines into the Docker image and expect them to work on every JetPack patch
release.  Instead we bundle the source `.pt` weights and regenerate engines
on first container start.

The regen takes ~60 sec per model on Orin Nano.  Subsequent starts skip it.

Usage:
    python3 regen_engines.py /app/alpr
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def needs_regen(pt_path: Path, engine_path: Path) -> bool:
    """Engine is out of date if it doesn't exist, OR is older than the .pt file."""
    if not engine_path.exists():
        return True
    return engine_path.stat().st_mtime < pt_path.stat().st_mtime


# YOLO11 weights this size or smaller will reliably TRT-compile on the
# Orin Nano 8 GB (verified with the YOLO11-nano plate detector at 5 MB).
# Anything bigger (e.g. YOLO11-L at 50 MB) needs ~1+ GB of GPU workspace
# during engine optimization, which doesn't fit alongside the rest of the
# container's CUDA context on a 7.6 GiB-visible GPU.  Such models still
# load fine from .pt at runtime, just slower per frame.  Larger devices
# (Orin NX, AGX Orin) can override this with an env var.
MAX_PT_SIZE_MB = int(os.environ.get("REGEN_ENGINES_MAX_PT_MB", "20"))


def regen(pt_path: Path) -> bool:
    """Use ultralytics to export `.pt` → `.engine`. Returns True on success."""
    size_mb = pt_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_PT_SIZE_MB:
        print(f"[regen_engines] SKIP {pt_path.name} "
              f"({size_mb:.1f} MB > MAX_PT_SIZE_MB={MAX_PT_SIZE_MB}); "
              f"will fall back to .pt at runtime (slower).  "
              f"Override with REGEN_ENGINES_MAX_PT_MB=<n> on larger Jetsons.")
        return True   # not a failure — intentional skip
    print(f"[regen_engines] exporting {pt_path.name} → TensorRT engine "
          f"(this takes ~60s on Orin Nano)")
    try:
        from ultralytics import YOLO   # local import — heavy
        model = YOLO(str(pt_path))
        # half=True for FP16 saves ~40% GPU memory + ~20% latency on Orin
        # with negligible accuracy loss for plate detection.
        model.export(format="engine", half=True, device=0, verbose=False)
    except Exception as e:    # noqa: BLE001
        print(f"[regen_engines] ERROR exporting {pt_path.name}: {e}", file=sys.stderr)
        return False
    return True


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: regen_engines.py <alpr_dir>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    if not root.is_dir():
        print(f"[regen_engines] {root} is not a directory; skipping", file=sys.stderr)
        return 0

    pt_files = sorted(root.glob("*.pt"))
    if not pt_files:
        print(f"[regen_engines] no .pt files in {root}; nothing to do")
        return 0

    failures = 0
    for pt in pt_files:
        engine = pt.with_suffix(".engine")
        if not needs_regen(pt, engine):
            print(f"[regen_engines] up-to-date: {engine.name}")
            continue
        if not regen(pt):
            failures += 1
    if failures:
        print(f"[regen_engines] {failures} engine(s) failed to build")
        return 1
    print("[regen_engines] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
