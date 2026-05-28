#!/usr/bin/env python3
"""Indian-plate ALPR runner.

Usage:
    python alpr.py --source csi                       # live IMX219 camera
    python alpr.py --source path/to/video.mp4         # video file
    python alpr.py --source path/to/photo.jpg         # single image
    python alpr.py --source path/to/folder/           # all images in folder

Common flags:
    --model PATH       detector .pt or .engine (default: yolov8x-oiv7.engine)
    --conf  FLOAT      detector confidence threshold (default 0.20)
    --db    PATH       SQLite store (default: ~/alpr/runs/plates.db)
    --save-crops       write each detected plate crop to ~/alpr/runs/crops/
    --headless         no preview window; useful for batching
    --no-track         skip ByteTrack + voting (one-shot OCR per frame)
    --vote-frames N    after this many frames per track, finalize the text (default 5)

Press 'q' in the preview window to quit live modes.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2

# Local modules
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from core import ALPRPipeline, PlateRead     # noqa: E402
from db import Row, Store                    # noqa: E402


# -------------------------------------------------------------------- sources

CAM_WIDTH, CAM_HEIGHT, CAM_FPS = 1280, 720, 30
FLIP_METHOD = 2


def csi_pipeline(sensor_id=0, w=CAM_WIDTH, h=CAM_HEIGHT, fps=CAM_FPS, flip=FLIP_METHOD):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip} ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=1 max-buffers=2"
    )


def open_source(spec: str):
    """Return (frames_iter, source_label, is_live).

    frames_iter yields (frame_idx, bgr_frame).  is_live=True means "infinite"
    stream — the runner should not announce 'done' at the end.
    """
    if spec == "csi":
        for sid in (0, 1):
            cap = cv2.VideoCapture(csi_pipeline(sensor_id=sid), cv2.CAP_GSTREAMER)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    return _cv2_iter(cap), f"csi:sensor{sid}", True
                cap.release()
        raise RuntimeError("Could not open IMX219.  See `v4l2-ctl --list-devices`.")

    p = Path(spec).expanduser()
    if not p.exists():
        raise FileNotFoundError(spec)

    if p.is_dir():
        imgs = sorted(p.glob("*.jpg")) + sorted(p.glob("*.jpeg")) + sorted(p.glob("*.png"))
        if not imgs:
            raise FileNotFoundError(f"no images in {p}")
        def gen():
            for i, f in enumerate(imgs):
                img = cv2.imread(str(f))
                if img is not None:
                    yield i, img
        return gen(), f"folder:{p.name}", False

    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"cv2 could not read {p}")
        return iter([(0, img)]), f"image:{p.name}", False

    # assume video
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {p}")
    return _cv2_iter(cap), f"video:{p.name}", False


def _cv2_iter(cap):
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            return
        yield i, frame
        i += 1


# ------------------------------------------------------------------- tracking

class TrackVoter:
    """Per-track-id OCR text voter.

    For each track_id we keep a small deque of (clean_text, ocr_conf, valid)
    tuples.  After `vote_frames` reads we finalize: majority vote on text,
    preferring valid-regex reads.
    """

    def __init__(self, vote_frames: int = 5):
        self.vote_frames = vote_frames
        self.history: dict[int, deque] = defaultdict(lambda: deque(maxlen=vote_frames * 2))
        self.finalized: dict[int, str] = {}

    def record(self, track_id: int, read: PlateRead) -> str | None:
        if not read.text:
            return self.finalized.get(track_id)
        self.history[track_id].append((read.text, read.ocr_conf, read.valid))

        # finalize once we have enough samples
        if len(self.history[track_id]) >= self.vote_frames:
            valids = [t for t, c, v in self.history[track_id] if v]
            pool = valids if valids else [t for t, c, v in self.history[track_id]]
            text, _ = Counter(pool).most_common(1)[0]
            self.finalized[track_id] = text
            return text
        return self.finalized.get(track_id)


# ------------------------------------------------------------------- main

def draw_overlay(frame, reads: list[PlateRead], voted: dict[int, str], fps: float):
    for r in reads:
        x1, y1, x2, y2 = r.bbox
        color = (0, 200, 0) if r.valid else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = r.text
        if r.track_id is not None and r.track_id in voted:
            label = f"#{r.track_id}: {voted[r.track_id]}"
        elif r.track_id is not None:
            label = f"#{r.track_id}: {r.text or '...'}"
        if label:
            cv2.putText(frame, label, (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    text = f"{fps:5.1f} FPS  plates:{len(reads)}"
    cv2.putText(frame, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="'csi', or path to video/image/folder")
    ap.add_argument("--model",  default="/home/jetson/alpr/license-plate-finetune-v1l.engine",
                    help="detector (.pt or .engine). Default: dedicated YOLO11-L "
                         "plate detector from morsetechlab/yolov11-license-plate-detection.")
    ap.add_argument("--ocr",    default="fast", choices=["fast", "easyocr"],
                    help="OCR backend. 'fast' = fast-plate-ocr (default, ~6 ms, "
                         "plate-specific); 'easyocr' = EasyOCR (~80 ms, general).")
    ap.add_argument("--conf",   type=float, default=0.20)
    ap.add_argument("--imgsz",  type=int,   default=640)
    ap.add_argument("--db",     default=str(HERE / "runs" / "plates.db"))
    ap.add_argument("--save-crops", action="store_true")
    ap.add_argument("--crops-dir", default=str(HERE / "runs" / "crops"))
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--no-track", action="store_true")
    ap.add_argument("--vote-frames", type=int, default=5)
    args = ap.parse_args()

    print(f"  Loading ALPR pipeline (detector={args.model}, ocr={args.ocr})…")
    pipe = ALPRPipeline(detector_path=args.model, imgsz=args.imgsz,
                        det_conf=args.conf, ocr_backend=args.ocr)
    store = Store(args.db)
    voter = None if args.no_track else TrackVoter(vote_frames=args.vote_frames)
    if args.save_crops:
        Path(args.crops_dir).mkdir(parents=True, exist_ok=True)

    print(f"  Opening source: {args.source}")
    frames, source_label, is_live = open_source(args.source)
    print(f"  Source: {source_label}{' (live)' if is_live else ''}")

    win = f"ALPR — {source_label} — q to quit"
    if not args.headless:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    fps_ema = 0.0
    last_t = time.perf_counter()

    try:
        for idx, frame in frames:
            if voter is not None:
                # Use ultralytics' built-in tracking via the detector,
                # then OCR each tracked plate crop.
                tracks = pipe.detector.track(
                    frame, imgsz=args.imgsz, device=0,
                    half=pipe._is_engine,
                    conf=args.conf, verbose=False, persist=True,
                    tracker="bytetrack.yaml",
                )[0]
                reads: list[PlateRead] = []
                if tracks.boxes is not None:
                    for b in tracks.boxes:
                        cls = int(b.cls)
                        if pipe.plate_class_ids and cls not in pipe.plate_class_ids:
                            continue
                        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                        if (x2 - x1) * (y2 - y1) < pipe.min_plate_area_px:
                            continue
                        crop = frame[max(0, y1-4):y2+4, max(0, x1-4):x2+4]
                        clean, raw, ocr_conf = pipe._ocr_crop(crop)
                        fixed, valid = pipe._try_char_fixes(clean) if clean else (clean, False)
                        tid = int(b.id.item()) if b.id is not None else -1
                        reads.append(PlateRead(
                            text=fixed, raw_text=raw,
                            det_conf=float(b.conf), ocr_conf=ocr_conf,
                            bbox=(x1, y1, x2, y2), valid=valid, track_id=tid,
                        ))
            else:
                reads = pipe.detect_and_read(frame)

            # voting + DB
            voted: dict[int, str] = {}
            for r in reads:
                if voter is not None and r.track_id is not None and r.track_id >= 0:
                    txt = voter.record(r.track_id, r)
                    if txt is not None:
                        voted[r.track_id] = txt
                    crop_path = ""
                    if args.save_crops and r.text:
                        ts = int(time.time() * 1000)
                        crop_fname = f"{source_label}_t{r.track_id}_{ts}.jpg"
                        crop_path = str(Path(args.crops_dir) / crop_fname)
                        x1, y1, x2, y2 = r.bbox
                        cv2.imwrite(crop_path, frame[max(0, y1-4):y2+4, max(0, x1-4):x2+4])
                    store.upsert(Row(
                        source=source_label,
                        track_id=r.track_id,
                        plate_text=voted.get(r.track_id, r.text),
                        raw_text=r.raw_text,
                        det_conf=r.det_conf, ocr_conf=r.ocr_conf,
                        valid=r.valid, crop_path=crop_path,
                    ))
                elif r.text and r.valid:
                    # no-track mode: only log clearly valid reads
                    store.upsert(Row(
                        source=source_label,
                        track_id=hash(r.text) & 0xFFFFFF,
                        plate_text=r.text, raw_text=r.raw_text,
                        det_conf=r.det_conf, ocr_conf=r.ocr_conf,
                        valid=True,
                    ))

            # fps + display
            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            inst = 1.0 / dt if dt > 0 else 0.0
            fps_ema = 0.9 * fps_ema + 0.1 * inst if fps_ema else inst

            if not args.headless:
                draw_overlay(frame, reads, voted, fps_ema)
                cv2.imshow(win, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif idx % 30 == 0:
                print(f"  frame {idx}: {fps_ema:.1f} FPS, {len(reads)} plates, "
                      f"finalized so far: {len(voted)}")
    finally:
        if not args.headless:
            cv2.destroyAllWindows()

    print("\n  Last 10 plates in DB:")
    for row in store.recent(10):
        print(f"    {row}")
    print(f"\n  DB: {args.db}")


if __name__ == "__main__":
    main()
