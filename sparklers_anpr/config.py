"""Sparkler ANPR Studio — persisted runtime config.

Loads from ~/sparklers_anpr/config.yaml on startup; writes back when the user
saves from the Settings page.  Defaults match what the Jetson Orin Nano
8 GB can reliably run.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.yaml"
DEFAULT_DETECTOR = "/home/jetson/alpr/license-plate-finetune-v1n.engine"
ALPR_MODULE_DIR = "/home/jetson/alpr"   # for reusing core.py, db.py


@dataclass
class PipelineConfig:
    detector_path: str = DEFAULT_DETECTOR
    detector_conf: float = 0.20
    imgsz: int = 640
    ocr_backend: str = "fast"            # fast | easyocr | fast+vlm
    vlm_model: str = "microsoft/Florence-2-base"
    vlm_device: str = "cuda"             # cpu | cuda (use cuda after the GPU-VLM recipe)
    vote_frames: int = 5
    min_plate_area_px: int = 400
    country: str = "IN"                  # IN | UK | US | EU | XX (generic)

    # detectors available in the UI dropdown
    detectors_available: list[dict] = field(default_factory=lambda: [
        {"label": "YOLO11-n plate (8 MB · fastest)",
         "path": "/home/jetson/alpr/license-plate-finetune-v1n.engine"},
        {"label": "YOLO11-L plate (52 MB · most accurate)",
         "path": "/home/jetson/alpr/license-plate-finetune-v1l.engine"},
        {"label": "YOLOv8-OIV7-x (OIV7 fallback)",
         "path": "/home/jetson/yolov8/yolov8x-oiv7.engine"},
    ])

    ocr_backends_available: list[dict] = field(default_factory=lambda: [
        {"label": "fast-plate-ocr  (~6 ms · default)",          "id": "fast"},
        {"label": "PaddleOCR PP-OCR  (~30 ms · accurate)",      "id": "paddle"},
        {"label": "EasyOCR  (~80 ms · general-purpose)",        "id": "easyocr"},
        {"label": "fast + Florence-2 VLM (GPU, ~4 s/verify)",   "id": "fast+vlm"},
        {"label": "fast + Florence-2 VLM (CPU, ~35 s/verify)",  "id": "fast+vlm-cpu"},
        {"label": "paddle + Florence-2 VLM (GPU)",              "id": "paddle+vlm"},
    ])


@dataclass
class UIConfig:
    title: str = "Sparkler ANPR Studio"
    subtitle: str = "Sparkler Ultimate Number Plate Recognition System"
    theme: str = "dark"        # dark | light
    play_sound_on_detect: bool = True
    show_fps_overlay: bool = True
    keep_log_days: int = 30
    crops_max_gb: float = 4.0


@dataclass
class SourceConfig:
    """Currently active source (set from the dashboard)."""
    type: str = "csi"          # csi | usb | image | video | rtsp
    csi_sensor_id: int = 0
    usb_device: str = "/dev/video0"
    rtsp_url: str = ""
    flip_method: int = 2       # nvvidconv flip-method (CSI only)
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass
class Config:
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    source: SourceConfig = field(default_factory=SourceConfig)


def _merge(d: dict, schema: type):
    """Coerce loaded dict into dataclass with defaults filled in."""
    inst = schema()
    for k, v in d.items():
        if hasattr(inst, k):
            setattr(inst, k, v)
    return inst


def load() -> Config:
    if not CONFIG_PATH.exists():
        return Config()
    try:
        data = yaml.safe_load(CONFIG_PATH.read_text()) or {}
        return Config(
            pipeline=_merge(data.get("pipeline", {}), PipelineConfig),
            ui=_merge(data.get("ui", {}), UIConfig),
            source=_merge(data.get("source", {}), SourceConfig),
        )
    except Exception as e:    # noqa: BLE001
        print(f"WARN: could not read {CONFIG_PATH} ({e}); using defaults")
        return Config()


def save(cfg: Config) -> None:
    CONFIG_PATH.write_text(yaml.safe_dump({
        "pipeline": asdict(cfg.pipeline),
        "ui": asdict(cfg.ui),
        "source": asdict(cfg.source),
    }, sort_keys=False))
