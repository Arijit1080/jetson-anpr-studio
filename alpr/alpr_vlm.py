#!/usr/bin/env python3
"""ALPR with async VLM verification.

Sister of alpr.py.  Same source modes (csi / video / image / folder).  On top
of the fast detect + fast-plate-ocr path, every new stable track is also
submitted to a background Qwen2-VL-2B verifier.  When the VLM result arrives
(~1.5–3 s later) the displayed label and SQLite row both update.

The live preview stays at ~25–30 FPS — the VLM never blocks the main thread.

Usage:
    python alpr_vlm.py --source csi
    python alpr_vlm.py --source ~/footage/traffic.mp4
    python alpr_vlm.py --source ~/Desktop/numberPlates/n1.JPG
    python alpr_vlm.py --source ~/Desktop/numberPlates/ --headless

Optional flags:
    --no-vlm           skip the VLM thread (same behaviour as alpr.py)
    --trust-vlm        once VLM has answered, hide the fast-OCR text
    --vlm-model ID     pick a different VLM (default Qwen/Qwen2-VL-2B-Instruct)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from core import ALPRPipeline, PlateRead    # noqa: E402
from db import Row, Store                   # noqa: E402
from vlm_worker import VLMWorker            # noqa: E402

# Reuse the source helpers from alpr.py without re-defining
from alpr import open_source, TrackVoter, csi_pipeline   # noqa: E402, F401


def draw_overlay(frame, reads: list[PlateRead], voted: dict[int, str],
                 vlm: VLMWorker | None, trust_vlm: bool, fps: float):
    for r in reads:
        x1, y1, x2, y2 = r.bbox
        fast = voted.get(r.track_id, r.text) if r.track_id is not None else r.text
        vlm_res = vlm.get(r.track_id) if (vlm and r.track_id is not None) else None
        vlm_text = vlm_res.text if vlm_res else None

        # decide colour + label
        if vlm_text:
            color = (0, 200, 0)
            if trust_vlm:
                label = f"#{r.track_id}: {vlm_text}"
            else:
                label = f"#{r.track_id}: {fast} | VLM: {vlm_text}"
        elif r.valid:
            color = (0, 200, 200)  # cyan — fast says it's a valid plate, VLM not back yet
            label = f"#{r.track_id}: {fast} (verifying...)"
        else:
            color = (0, 165, 255)
            label = f"#{r.track_id}: {fast or '...'}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if label:
            cv2.putText(frame, label, (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    pending = vlm.pending_count() if vlm else 0
    txt = f"{fps:5.1f} FPS  plates:{len(reads)}  vlm-queue:{pending}"
    cv2.putText(frame, txt, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, txt, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--model", default="/home/jetson/alpr/license-plate-finetune-v1l.engine")
    ap.add_argument("--ocr",   default="fast", choices=["fast", "easyocr"])
    ap.add_argument("--conf",  type=float, default=0.20)
    ap.add_argument("--imgsz", type=int,   default=640)
    ap.add_argument("--db",    default=str(HERE / "runs" / "plates_vlm.db"))
    ap.add_argument("--save-crops", action="store_true")
    ap.add_argument("--crops-dir", default=str(HERE / "runs" / "crops_vlm"))
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--no-track", action="store_true")
    ap.add_argument("--vote-frames", type=int, default=5)
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--trust-vlm", action="store_true")
    ap.add_argument("--vlm-model", default="microsoft/Florence-2-base",
                    help="VLM for verification. Default: Florence-2-base "
                         "(~500 MB, fast). Alternatives: microsoft/Florence-2-large, "
                         "Qwen/Qwen2-VL-2B-Instruct (needs >5 GB free GPU).")
    ap.add_argument("--vlm-device", default="cpu", choices=["cpu", "cuda"],
                    help="Where to run the VLM. Default 'cpu' on Orin Nano 8 GB "
                         "(GPU OOMs alongside the YOLO TRT engine). Use 'cuda' on "
                         "Orin NX 16 GB / AGX Orin 32-64 GB.")
    ap.add_argument("--vlm-wait", action="store_true",
                    help="block at end for any in-flight VLM jobs to complete "
                         "(useful for single-image / batch modes)")
    args = ap.parse_args()

    print(f"  Loading ALPR pipeline (detector={args.model}, ocr={args.ocr})…")
    pipe = ALPRPipeline(detector_path=args.model, imgsz=args.imgsz,
                        det_conf=args.conf, ocr_backend=args.ocr)
    store = Store(args.db)
    voter = None if args.no_track else TrackVoter(vote_frames=args.vote_frames)

    vlm: VLMWorker | None = None
    if not args.no_vlm:
        vlm = VLMWorker(model_id=args.vlm_model, device=args.vlm_device)
        vlm.start()
        print(f"  VLM worker spawned ({args.vlm_model} on {args.vlm_device}, "
              "loading in background, ~30-60s on first run)…")

    if args.save_crops:
        Path(args.crops_dir).mkdir(parents=True, exist_ok=True)

    print(f"  Opening source: {args.source}")
    frames, source_label, is_live = open_source(args.source)
    print(f"  Source: {source_label}{' (live)' if is_live else ''}")

    win = f"ALPR+VLM — {source_label} — q to quit"
    if not args.headless:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    fps_ema = 0.0
    last_t = time.perf_counter()

    try:
        for idx, frame in frames:
            # ----- detect + track + fast OCR (same as alpr.py) -----
            if voter is not None:
                tracks = pipe.detector.track(
                    frame, imgsz=args.imgsz, device=0,
                    half=pipe._is_engine, conf=args.conf,
                    verbose=False, persist=True, tracker="bytetrack.yaml",
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

            # ----- voting, VLM submit, DB update -----
            voted: dict[int, str] = {}
            for r in reads:
                if voter is not None and r.track_id is not None and r.track_id >= 0:
                    txt = voter.record(r.track_id, r)
                    if txt is not None:
                        voted[r.track_id] = txt
                    # submit to VLM once the fast track is stable enough to crop
                    if vlm is not None and r.text:
                        x1, y1, x2, y2 = r.bbox
                        crop = frame[max(0, y1-4):y2+4, max(0, x1-4):x2+4]
                        vlm.submit(r.track_id, crop)

                    final_text = voted.get(r.track_id, r.text)
                    vlm_res = vlm.get(r.track_id) if vlm else None
                    if vlm_res:
                        final_text = vlm_res.text
                    if r.text:
                        crop_path = ""
                        if args.save_crops:
                            ts = int(time.time() * 1000)
                            crop_fname = f"{source_label.replace('/', '_')}_t{r.track_id}_{ts}.jpg"
                            crop_path = str(Path(args.crops_dir) / crop_fname)
                            x1, y1, x2, y2 = r.bbox
                            cv2.imwrite(crop_path, frame[max(0, y1-4):y2+4, max(0, x1-4):x2+4])
                        store.upsert(Row(
                            source=source_label, track_id=r.track_id,
                            plate_text=final_text, raw_text=r.raw_text,
                            det_conf=r.det_conf, ocr_conf=r.ocr_conf,
                            valid=r.valid or bool(vlm_res),
                            crop_path=crop_path,
                        ))

            # ----- fps + display -----
            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            inst = 1.0 / dt if dt > 0 else 0.0
            fps_ema = 0.9 * fps_ema + 0.1 * inst if fps_ema else inst

            if not args.headless:
                draw_overlay(frame, reads, voted, vlm, args.trust_vlm, fps_ema)
                cv2.imshow(win, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif idx % 30 == 0:
                print(f"  frame {idx}: {fps_ema:.1f} FPS, {len(reads)} plates, "
                      f"vlm-pending {vlm.pending_count() if vlm else 0}")

        # For single-image / batch sources, optionally wait for VLM jobs.
        if vlm is not None and (args.vlm_wait or not is_live):
            submitted = sorted(vlm._submitted)
            if submitted:
                print(f"\n  Waiting for VLM to finish {len(submitted)} plate(s) "
                      "(Florence-2 on CPU = ~25 s per plate)…")
                deadline = time.time() + 60.0 * len(submitted)   # generous per-plate budget
                pending = list(submitted)
                while pending and time.time() < deadline:
                    pending = [t for t in pending if vlm.get(t) is None]
                    if pending:
                        time.sleep(0.5)
                # write whatever results we got back to the DB
                with vlm._lock:
                    for tid, res in vlm._results.items():
                        store.upsert(Row(
                            source=source_label, track_id=tid,
                            plate_text=res.text, raw_text=res.raw,
                            det_conf=0.0, ocr_conf=1.0, valid=True,
                        ))
                print(f"  VLM done: {len(vlm._results)}/{len(submitted)} verified")
    finally:
        if not args.headless:
            cv2.destroyAllWindows()
        if vlm is not None:
            vlm.stop()

    print("\n  Last 10 plates in DB:")
    for row in store.recent(10):
        print(f"    {row}")
    print(f"\n  DB: {args.db}")


if __name__ == "__main__":
    main()
