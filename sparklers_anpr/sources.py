"""Sparkler ANPR Studio — input source adapters.

A source produces BGR frames on demand via `iter_frames()`.  Five kinds:
- CSI camera via nvarguscamerasrc (Jetson IMX219 etc.)
- USB camera via V4L2
- Image file (single frame; iterator yields once then exits)
- Video file
- RTSP / HTTP MJPEG URL
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2


def csi_pipeline(sensor_id: int = 0,
                 width: int = 1280, height: int = 720,
                 framerate: int = 30, flip_method: int = 2) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},"
        f"framerate={framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        "video/x-raw,format=BGRx ! videoconvert ! "
        "video/x-raw,format=BGR ! appsink drop=1 max-buffers=2"
    )


@dataclass
class SourceSpec:
    type: str                       # csi | usb | image | video | rtsp
    uri: str = ""                   # path/url/device; "" for csi
    csi_sensor_id: int = 0
    flip_method: int = 2
    width: int = 1280
    height: int = 720
    fps: int = 30


class Source:
    """Wraps a cv2.VideoCapture (or a static image) and exposes frames."""

    def __init__(self, spec: SourceSpec):
        self.spec = spec
        self.cap: Optional[cv2.VideoCapture] = None
        self._static_image: Optional["np.ndarray"] = None  # type: ignore
        self._open()

    # ---- lifecycle ----------------------------------------------------------

    def _open(self) -> None:
        s = self.spec
        if s.type == "csi":
            pipe = csi_pipeline(s.csi_sensor_id, s.width, s.height, s.fps, s.flip_method)
            self.cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
            # Try sensor 1 if sensor 0 fails
            if not self.cap.isOpened() or not self.cap.read()[0]:
                if self.cap is not None:
                    self.cap.release()
                pipe = csi_pipeline(1, s.width, s.height, s.fps, s.flip_method)
                self.cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        elif s.type == "usb":
            dev = s.uri or "/dev/video0"
            # If a number was passed like "0", treat as cam index
            cam_idx = int(dev) if dev.isdigit() else dev
            self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, s.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, s.height)
            self.cap.set(cv2.CAP_PROP_FPS, s.fps)
        elif s.type == "rtsp":
            self.cap = cv2.VideoCapture(s.uri, cv2.CAP_FFMPEG)
        elif s.type == "video":
            self.cap = cv2.VideoCapture(s.uri)
        elif s.type == "image":
            img = cv2.imread(s.uri)
            if img is None:
                raise RuntimeError(f"cv2 could not read image: {s.uri}")
            self._static_image = img
        else:
            raise ValueError(f"unknown source type {s.type!r}")

        if self.cap is not None and not self.cap.isOpened():
            raise RuntimeError(f"could not open source {s.type}:{s.uri}")

    def is_live(self) -> bool:
        return self.spec.type in ("csi", "usb", "rtsp")

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._static_image = None

    # ---- frame iteration ----------------------------------------------------

    def iter_frames(self) -> Iterator[tuple[int, "np.ndarray"]]:    # type: ignore
        """Yield (frame_idx, BGR frame).  Stops on EOF for non-live sources."""
        if self._static_image is not None:
            yield 0, self._static_image
            return
        if self.cap is None:
            return
        i = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                # video file ended; live sources should retry once
                if self.is_live():
                    time.sleep(0.05)
                    continue
                return
            yield i, frame
            i += 1


# ----- helpers -------------------------------------------------------------

def list_usb_cameras() -> list[str]:
    """Best-effort enumeration of /dev/video* that look like real cams."""
    out = []
    for p in sorted(Path("/dev").glob("video*")):
        try:
            cap = cv2.VideoCapture(str(p), cv2.CAP_V4L2)
            if cap.isOpened():
                out.append(str(p))
            cap.release()
        except Exception:    # noqa: BLE001
            pass
    return out
