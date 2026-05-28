"""Bake the OCR + VLM model weights into the Docker image at build time.

Without this, the first container start would download ~300 MB on the user's
network — and silently fail if the host is offline.  Running this during
`docker build` means weights end up in image layers that ship to every
machine that pulls the image.

Downloads:
    - Florence-2-base (HuggingFace, ~270 MB)
    - PaddleOCR PP-OCRv4 English det + rec + cls (~16 MB)
    - fast-plate-ocr cct-xs-v1-global-model (~2 MB)

Targets:
    HF cache    → /opt/models/huggingface
    Paddle      → /opt/models/paddleocr
    fast-plate  → /opt/models/fast_plate_ocr
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


# ---- routes models to /opt/models so the docker-compose volume preserves
#      them across container rebuilds ----------------------------------------
HF_HOME = os.environ.setdefault("HF_HOME", "/opt/models/huggingface")
PADDLE_BASE = os.environ.setdefault("PADDLE_OCR_BASE_DIR", "/opt/models/paddleocr")
FAST_BASE = os.environ.setdefault("FAST_PLATE_OCR_HOME", "/opt/models/fast_plate_ocr")
for d in (HF_HOME, PADDLE_BASE, FAST_BASE):
    Path(d).mkdir(parents=True, exist_ok=True)


def _section(name: str) -> None:
    print(f"\n{'='*60}\n {name}\n{'='*60}")


# ---------------------------------------------------------------------------
# Florence-2 (microsoft/Florence-2-base)
# ---------------------------------------------------------------------------
def fetch_florence2() -> None:
    _section("Florence-2-base (~270 MB)")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="microsoft/Florence-2-base",
            allow_patterns=[
                "*.json", "*.txt", "*.py", "*.safetensors", "*.bin",
                "tokenizer*", "preprocessor*", "processor*"
            ],
            cache_dir=HF_HOME,
        )
        print("✓ Florence-2-base cached")
    except Exception as e:    # noqa: BLE001
        print(f"WARN: Florence-2 download failed: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# PaddleOCR PP-OCRv4 English (det + rec + cls)
# ---------------------------------------------------------------------------
def fetch_paddle() -> None:
    _section("PaddleOCR PP-OCRv4 en (~16 MB)")
    try:
        # paddleocr 2.7 auto-downloads to ~/.paddleocr/whl/ on first init.
        # Redirect via HOME so it lands under /opt/models/paddleocr.
        os.environ["HOME"] = PADDLE_BASE
        Path(f"{PADDLE_BASE}/.paddleocr").mkdir(parents=True, exist_ok=True)

        from paddleocr import PaddleOCR
        _ = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            show_log=False,
        )
        print("✓ PaddleOCR weights cached")
    except Exception as e:    # noqa: BLE001
        print(f"WARN: PaddleOCR download failed: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# fast-plate-ocr cct-xs-v1-global-model
# ---------------------------------------------------------------------------
def fetch_fast_plate() -> None:
    _section("fast-plate-ocr cct-xs-v1-global-model (~2 MB)")
    try:
        # fast-plate-ocr 0.3+ caches under ~/.cache by default.  No env
        # override — just let it write to the standard location and we
        # capture it by mounting HOME.
        from fast_plate_ocr import LicensePlateRecognizer
        _ = LicensePlateRecognizer("cct-xs-v1-global-model")
        print("✓ fast-plate-ocr weights cached")
    except Exception as e:    # noqa: BLE001
        print(f"WARN: fast-plate-ocr download failed: {e}")
        traceback.print_exc()


def main() -> int:
    fetch_florence2()
    fetch_paddle()
    fetch_fast_plate()
    print("\n✓ prefetch complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
