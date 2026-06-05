"""Sparkler ANPR Studio — pipeline runner.

A long-lived background worker that:
- pulls frames from a `Source`
- runs the ALPR pipeline (detect → OCR → optional VLM verify)
- writes detections to the SQLite store
- broadcasts events to SSE subscribers
- exposes the most recent annotated JPEG for the MJPEG stream

The actual detect+OCR logic lives in /home/jetson/alpr/core.py — we just
reuse `ALPRPipeline` to avoid duplicating work.  VLM verification is wired
through /home/jetson/alpr/vlm_worker.py.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2

# Reuse the existing ALPR core + VLM worker
ALPR_DIR = "/home/jetson/alpr"
if ALPR_DIR not in sys.path:
    sys.path.insert(0, ALPR_DIR)
from core import ALPRPipeline, PlateRead   # noqa: E402
from vlm_worker import VLMWorker           # noqa: E402

from config import Config, PipelineConfig, SourceConfig    # noqa: E402
from countries import get_profile                          # noqa: E402
from db import Plate, Store                                # noqa: E402
from sources import Source, SourceSpec                     # noqa: E402

# ---------- VLM crop preprocessing -----------------------------------------

VLM_PAD_PCT = 0.20             # pad bbox 20% on each side before sending to VLM
VLM_VOTES_PER_TRACK = 3        # how many VLM submissions per track for voting
VLM_SUB_INTERVAL_FRAMES = 3    # submit a new VLM crop every Nth frame for a track


def _vlm_crop(frame, x1: int, y1: int, x2: int, y2: int,
              pad_pct: float = VLM_PAD_PCT):
    """Extract a padded + CLAHE-contrast-enhanced crop for VLM input.

    Florence-2 reads better when:
      a) the crop has 10-20% margin around the plate edges (tight YOLO bboxes
         clip the leftmost/rightmost characters);
      b) low-contrast plates (shaded, bright sun) get histogram-equalised.
    """
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad_pct), int(bh * pad_pct)
    nx1 = max(0, x1 - px); ny1 = max(0, y1 - py)
    nx2 = min(w, x2 + px); ny2 = min(h, y2 + py)
    crop = frame[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        return crop
    # CLAHE on the L channel of LAB (boosts shadow detail without blowing highlights)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------- events bus

class EventBus:
    """Thread-safe fan-out for SSE clients.  Pipeline (sync thread) pushes
    events; the FastAPI event loop drains per-client queues."""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._clients: list[asyncio.Queue] = []
        self._lock = threading.Lock()
        self._history: deque[dict] = deque(maxlen=200)

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        with self._lock:
            self._clients.append(q)
            # send history so newcomers see recent events
            for e in list(self._history)[-30:]:
                try:
                    q.put_nowait(e)
                except asyncio.QueueFull:
                    pass
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            try:
                self._clients.remove(q)
            except ValueError:
                pass

    def publish(self, event: dict) -> None:
        """Called from any thread.  Drops events if the FastAPI loop isn't
        attached yet (boot ordering)."""
        with self._lock:
            self._history.append(event)
            clients = list(self._clients)
        if self._loop is None:
            return
        for q in clients:
            try:
                self._loop.call_soon_threadsafe(q.put_nowait, event)
            except RuntimeError:
                pass


# ---------------------------------------------------------------- runner

@dataclass
class RunnerStatus:
    running: bool = False
    source_type: str = ""
    source_uri: str = ""
    backend: str = "fast"
    detector: str = ""
    session_id: str = ""
    fps: float = 0.0
    plates_in_session: int = 0
    vlm_queue: int = 0
    vlm_device: str = "cpu"
    last_error: str = ""


class PipelineRunner:
    """Drives the ALPR pipeline against a Source on a background thread."""

    def __init__(self, store: Store, bus: EventBus, runs_dir: Path):
        self.store = store
        self.bus = bus
        self.runs_dir = runs_dir
        (self.runs_dir / "crops").mkdir(parents=True, exist_ok=True)
        (self.runs_dir / "frames").mkdir(parents=True, exist_ok=True)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        # Per-session pointers — populated by _build_pipeline, cleared by stop()
        self._pipeline: Optional[ALPRPipeline] = None
        self._vlm: Optional[VLMWorker] = None
        # Process-level model singletons — survive across sessions so we don't
        # pay the 5–10s reload cost on every /api/start.  Keyed on the bits of
        # config that determine *which* model gets loaded; cache misses evict
        # the old one before loading new (avoids GPU OOM on 8 GB Orin Nano).
        self._cached_pipeline: Optional[ALPRPipeline] = None
        self._cached_pipeline_key: Optional[tuple] = None
        self._cached_vlm: Optional[VLMWorker] = None
        self._cached_vlm_key: Optional[tuple] = None
        self._source: Optional[Source] = None
        self._latest_jpg: Optional[bytes] = None
        self._jpg_lock = threading.Lock()
        self.status = RunnerStatus()

    # ----- public control surface -------------------------------------------

    def start(self, source_spec: SourceSpec, p_cfg: PipelineConfig) -> None:
        if self._thread is not None and self._thread.is_alive():
            self.stop()
        self._stop_event.clear()
        self.status.last_error = ""
        try:
            self._build_pipeline(p_cfg)
            self._source = Source(source_spec)
        except Exception as e:    # noqa: BLE001
            self.status.last_error = f"{type(e).__name__}: {e}"
            self.status.running = False
            self.bus.publish({"type": "error", "message": self.status.last_error})
            return

        session = self.store.new_session(
            source_type=source_spec.type,
            source_uri=source_spec.uri or (f"sensor-{source_spec.csi_sensor_id}"
                                            if source_spec.type == "csi" else ""),
            backend=p_cfg.ocr_backend,
            detector=p_cfg.detector_path,
        )
        self.status.session_id = session.id
        self.status.source_type = source_spec.type
        self.status.source_uri = source_spec.uri
        self.status.backend = p_cfg.ocr_backend
        self.status.detector = p_cfg.detector_path
        self.status.plates_in_session = 0
        self.status.running = True
        self.bus.publish({
            "type": "session_started",
            "session_id": session.id,
            "source_type": source_spec.type,
            "source_uri": source_spec.uri,
            "backend": p_cfg.ocr_backend,
            "detector": Path(p_cfg.detector_path).name,
        })

        self._thread = threading.Thread(target=self._run, args=(session,),
                                        name="ALPRRunner", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """End the current session.  The cached YOLO+OCR pipeline and the
        cached VLM worker are NOT destroyed — they stay warm so the next
        session starts in ~0s instead of ~8s.  Use `shutdown_models()` for
        a full release (called from the FastAPI shutdown hook)."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        self._thread = None
        if self._source is not None:
            self._source.release()
        self._source = None
        # Just clear the per-session pointers — the underlying singletons
        # live on under self._cached_pipeline / self._cached_vlm.
        self._pipeline = None
        self._vlm = None
        if self.status.session_id:
            self.store.end_session(self.status.session_id)
            self.bus.publish({"type": "session_ended",
                              "session_id": self.status.session_id})
        self.status.running = False

    def shutdown_models(self) -> None:
        """Fully release the cached models.  Called from the FastAPI
        `shutdown` event so the process exits cleanly.  Don't call this
        between sessions — that's what stop() is for."""
        if self._cached_vlm is not None:
            try:
                self._cached_vlm.stop()
            except Exception:    # noqa: BLE001
                pass
            self._cached_vlm = None
            self._cached_vlm_key = None
        self._cached_pipeline = None
        self._cached_pipeline_key = None
        try:
            import gc; gc.collect()
            import torch
            torch.cuda.empty_cache()
        except Exception:    # noqa: BLE001
            pass

    def latest_jpg(self) -> Optional[bytes]:
        with self._jpg_lock:
            return self._latest_jpg

    # ----- internals --------------------------------------------------------

    def _build_pipeline(self, cfg: PipelineConfig) -> None:
        # Map UI-friendly backend ids → (ocr_backend, use_vlm, vlm_device).
        # Supported ids:  fast, paddle, easyocr,
        #                 fast+vlm, fast+vlm-cpu, paddle+vlm, paddle+vlm-cpu, easyocr+vlm…
        # +vlm     → CUDA VLM        +vlm-cpu → CPU VLM
        backend = cfg.ocr_backend
        if backend.endswith("+vlm-cpu"):
            use_vlm = True; vlm_device = "cpu"
            base = backend[:-len("+vlm-cpu")]
        elif backend.endswith("+vlm"):
            use_vlm = True; vlm_device = "cuda"
            base = backend[:-len("+vlm")]
        else:
            use_vlm = False; vlm_device = ""
            base = backend
        ocr_backend = base if base in ("fast", "easyocr", "paddle") else "fast"
        self._country_code = cfg.country

        # ----- YOLO+OCR pipeline: reuse if the config matches the cache -----
        # Cache key includes only the inputs that affect which model gets
        # loaded.  vote_frames / country / vlm_* don't change the loaded
        # weights and so don't invalidate the cache.
        pipe_key = (cfg.detector_path, cfg.imgsz, cfg.detector_conf,
                    ocr_backend, cfg.min_plate_area_px)
        if self._cached_pipeline_key != pipe_key:
            if self._cached_pipeline is not None:
                # Release old before loading new so we don't double-allocate
                # GPU memory on the 8 GB Orin Nano.
                self._cached_pipeline = None
                try:
                    import gc; gc.collect()
                    import torch
                    torch.cuda.empty_cache()
                except Exception:    # noqa: BLE001
                    pass
            self._cached_pipeline = ALPRPipeline(
                detector_path=cfg.detector_path,
                imgsz=cfg.imgsz,
                det_conf=cfg.detector_conf,
                ocr_backend=ocr_backend,
                min_plate_area_px=cfg.min_plate_area_px,
            )
            self._cached_pipeline_key = pipe_key
        self._pipeline = self._cached_pipeline

        # ----- VLM worker: reuse if model+device matches the cache -----
        if use_vlm:
            vlm_key = (cfg.vlm_model, vlm_device)
            if self._cached_vlm_key != vlm_key:
                # Different model or device requested → tear down old VLM
                # first to free the ~1.5 GB it's holding on the GPU before
                # loading a new one (Florence-2-base alone is ~270 MB on
                # disk but balloons to ~1.4 GB once loaded to fp16 CUDA).
                if self._cached_vlm is not None:
                    try:
                        self._cached_vlm.stop()
                    except Exception:    # noqa: BLE001
                        pass
                    self._cached_vlm = None
                    self._cached_vlm_key = None
                    try:
                        import gc; gc.collect()
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:    # noqa: BLE001
                        pass
                self._cached_vlm = VLMWorker(model_id=cfg.vlm_model,
                                              device=vlm_device)
                self._cached_vlm.start()
                self._cached_vlm_key = vlm_key
            else:
                # Same VLM as last time → just clear its per-session result
                # map so new track IDs aren't rejected as duplicates.
                self._cached_vlm.reset_session()
            self._vlm = self._cached_vlm
            self.status.vlm_device = vlm_device
        else:
            # No VLM this session.  We keep the cached one in memory in case
            # the next session re-enables it — avoids the ~8s reload.
            self._vlm = None
            self.status.vlm_device = ""

    def _run(self, session) -> None:
        pipe = self._pipeline
        vlm = self._vlm
        source = self._source
        if pipe is None or source is None:
            return
        country = get_profile(self._country_code)

        votes: dict[int, deque] = defaultdict(lambda: deque(maxlen=10))
        finalized: dict[int, str] = {}
        crop_saved: set[int] = set()
        vlm_db_updated: set[int] = set()       # tracks already synced to DB after VLM verify

        # ---- multi-frame VLM voting state ----
        # Submit up to VLM_VOTES_PER_TRACK crops per track (spaced VLM_SUB_INTERVAL_FRAMES apart).
        # Each goes in with a synthetic id = track_id * 1000 + sub_index so we can collect them.
        vlm_sub_count: dict[int, int] = defaultdict(int)     # tid -> # submitted
        vlm_frames_seen: dict[int, int] = defaultdict(int)   # tid -> # frames this track has been in
        vlm_synth_ids: dict[int, list[int]] = defaultdict(list)  # tid -> [synthetic ids]
        vlm_synth_done: set[int] = set()                     # synthetic ids whose result we've read
        vlm_votes: dict[int, list[str]] = defaultdict(list)  # tid -> texts voted by VLM

        fps_ema = 0.0
        last_t = time.perf_counter()
        live = source.is_live()

        try:
            for idx, frame in source.iter_frames():
                if self._stop_event.is_set():
                    break

                # --- detect + OCR + ByteTrack (call ultralytics .track) ---
                tracks = pipe.detector.track(
                    frame, imgsz=pipe.imgsz, device=0,
                    half=pipe._is_engine, conf=pipe.det_conf,
                    verbose=False, persist=True, tracker="bytetrack.yaml",
                )[0]
                reads: list[tuple[PlateRead, "np.ndarray"]] = []   # type: ignore
                if tracks.boxes is not None:
                    for b in tracks.boxes:
                        cls = int(b.cls)
                        if pipe.plate_class_ids and cls not in pipe.plate_class_ids:
                            continue
                        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                        if (x2 - x1) * (y2 - y1) < pipe.min_plate_area_px:
                            continue
                        crop = frame[max(0, y1-4):y2+4, max(0, x1-4):x2+4]
                        _clean, raw, ocr_conf = pipe._ocr_crop(crop)
                        # New simplified logic: just country.clean() — NO regex
                        # validation.  Whatever the OCR/VLM agree on is "verified".
                        c_text = country.clean(raw)
                        tid = int(b.id.item()) if b.id is not None else -1
                        pr = PlateRead(text=c_text, raw_text=raw,
                                       det_conf=float(b.conf), ocr_conf=ocr_conf,
                                       bbox=(x1, y1, x2, y2), valid=False,
                                       track_id=tid)
                        reads.append((pr, crop))

                # --- vote + VLM submit + DB upsert -----------------------
                for pr, crop in reads:
                    if pr.text and pr.track_id is not None and pr.track_id >= 0:
                        votes[pr.track_id].append(pr.text)
                        if len(votes[pr.track_id]) >= 5:
                            text, _ = Counter(votes[pr.track_id]).most_common(1)[0]
                            finalized[pr.track_id] = text
                        # multi-frame VLM submission: pad bbox + CLAHE,
                        # submit up to VLM_VOTES_PER_TRACK crops spaced out
                        if vlm is not None and pr.text:
                            vlm_frames_seen[pr.track_id] += 1
                            already = vlm_sub_count[pr.track_id]
                            if already < VLM_VOTES_PER_TRACK:
                                first = (already == 0)
                                spaced = (vlm_frames_seen[pr.track_id] % VLM_SUB_INTERVAL_FRAMES == 0)
                                if first or spaced:
                                    sid = pr.track_id * 1000 + already
                                    padded = _vlm_crop(frame, pr.bbox[0], pr.bbox[1],
                                                        pr.bbox[2], pr.bbox[3])
                                    if padded.size > 0 and vlm.submit(sid, padded):
                                        vlm_synth_ids[pr.track_id].append(sid)
                                        vlm_sub_count[pr.track_id] = already + 1

                        # Aggregate any VLM votes that have arrived for this track
                        vlm_res = None
                        if vlm is not None:
                            for sid in vlm_synth_ids.get(pr.track_id, []):
                                if sid in vlm_synth_done:
                                    continue
                                r = vlm.get(sid)
                                if r:
                                    vlm_votes[pr.track_id].append(country.clean(r.raw))
                                    vlm_synth_done.add(sid)
                            if vlm_votes.get(pr.track_id):
                                voted_text, _ = Counter(vlm_votes[pr.track_id]).most_common(1)[0]
                                vlm_res = type("VR", (), {"text": voted_text})()
                        # Two-column model: store OCR text and VLM text separately.
                        # "verified" is computed: it's True iff both are non-empty and equal.
                        ocr_text = country.clean(finalized.get(pr.track_id, pr.text))
                        vlm_text = country.clean(vlm_res.text) if vlm_res else ""
                        verified = bool(vlm_text) and ocr_text == vlm_text

                        # save crop on FIRST sighting of every track
                        crop_path = ""
                        if pr.track_id not in crop_saved and crop.size > 0:
                            cdir = self.runs_dir / "crops" / session.id
                            cdir.mkdir(parents=True, exist_ok=True)
                            fname = f"{int(time.time()*1000)}_{pr.track_id}_{(ocr_text or 'unk')[:20]}.jpg"
                            crop_path = f"{session.id}/{fname}"
                            cv2.imwrite(str(cdir / fname), crop)
                            crop_saved.add(pr.track_id)

                        plate_id = self.store.upsert_plate(Plate(
                            session_id=session.id,
                            track_id=pr.track_id,
                            plate_text=ocr_text,
                            vlm_text=vlm_text,
                            raw_text=pr.raw_text,
                            det_conf=pr.det_conf,
                            ocr_conf=pr.ocr_conf,
                            crop_path=crop_path,
                        ))

                        # `final` is what we draw on the live frame label
                        final = vlm_text if vlm_text else ocr_text
                        final_vlm = bool(vlm_text)
                        final_valid = verified

                        self.bus.publish({
                            "type": "vlm_verified" if final_vlm else "plate",
                            "session_id": session.id,
                            "plate_id": plate_id,
                            "track_id": pr.track_id,
                            "plate_text": ocr_text,
                            "vlm_text": vlm_text,
                            "verified": verified,
                            "raw_text": pr.raw_text,
                            "det_conf": round(pr.det_conf, 2),
                            "ocr_conf": round(pr.ocr_conf, 2),
                            "crop_path": crop_path,
                            "ts": time.time(),
                        })

                # --- sync any newly-completed VLM votes to DB --------
                # Plates often leave the frame before the VLM thread finishes.
                # When all submissions for a track come back, vote and persist
                # the VLM text in its own column.  The DB row already has the
                # OCR text from earlier frames; we just add vlm_text now.
                if vlm is not None:
                    for tid, sids in list(vlm_synth_ids.items()):
                        if tid in vlm_db_updated:
                            continue
                        for sid in sids:
                            if sid in vlm_synth_done:
                                continue
                            r = vlm.get(sid)
                            if r:
                                vlm_votes[tid].append(country.clean(r.raw))
                                vlm_synth_done.add(sid)
                        completed = sum(1 for s in sids if s in vlm_synth_done)
                        if vlm_votes[tid] and completed == len(sids):
                            voted_text, _ = Counter(vlm_votes[tid]).most_common(1)[0]
                            self.store.upsert_plate(Plate(
                                session_id=session.id,
                                track_id=tid,
                                plate_text="",          # leave OCR text untouched
                                vlm_text=voted_text,
                                raw_text=" | ".join(vlm_votes[tid]),
                            ))
                            vlm_db_updated.add(tid)
                            self.bus.publish({
                                "type": "vlm_verified",
                                "session_id": session.id,
                                "track_id": tid,
                                "vlm_text": voted_text,
                                "raw_text": " | ".join(vlm_votes[tid]),
                                "votes": vlm_votes[tid],
                                "ts": time.time(),
                            })

                # --- annotate frame + cache JPEG for MJPEG stream --------
                ann = frame.copy()
                for pr, _ in reads:
                    x1, y1, x2, y2 = pr.bbox
                    ocr = country.clean(finalized.get(pr.track_id, pr.text))
                    vlm_t = ""
                    if pr.track_id in vlm_votes and vlm_votes[pr.track_id]:
                        vlm_t, _ = Counter(vlm_votes[pr.track_id]).most_common(1)[0]
                    if vlm_t and ocr == vlm_t:
                        col = (0, 200, 0);   lab = f"{ocr}  ✓"
                    elif vlm_t:
                        col = (0, 200, 200); lab = f"OCR:{ocr}  VLM:{vlm_t}"
                    else:
                        col = (0, 165, 255); lab = ocr or "..."
                    cv2.rectangle(ann, (x1, y1), (x2, y2), col, 2)
                    cv2.putText(ann, lab, (x1, max(20, y1-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

                # FPS + overlay
                now = time.perf_counter()
                dt = now - last_t
                last_t = now
                inst = 1.0 / dt if dt > 0 else 0.0
                fps_ema = 0.9 * fps_ema + 0.1 * inst if fps_ema else inst
                txt = (f"{fps_ema:5.1f} FPS  plates:{len(reads)}  "
                       f"vlm-q:{vlm.pending_count() if vlm else 0}")
                cv2.putText(ann, txt, (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0,0,0), 5, cv2.LINE_AA)
                cv2.putText(ann, txt, (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0,255,255), 2, cv2.LINE_AA)

                ok, buf = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with self._jpg_lock:
                        self._latest_jpg = buf.tobytes()

                self.status.fps = fps_ema
                self.status.plates_in_session = self.store.stats()["total_detections"]
                self.status.vlm_queue = vlm.pending_count() if vlm else 0

                # for non-live, sleep a tiny bit so MJPEG can keep up
                if not live:
                    time.sleep(0.005)

            # ----- non-live source ended: drain in-flight VLM jobs --------
            # For image/video, the for-loop above exits before the slow Florence-2
            # results arrive.  Wait for every submitted synthetic id to come back,
            # then vote per original track and write to DB.
            if vlm is not None and not live and not self._stop_event.is_set():
                all_synth = [s for sids in vlm_synth_ids.values() for s in sids]
                if all_synth:
                    self.bus.publish({"type": "info",
                                      "message": f"Waiting for VLM to finish {len(all_synth)} verification(s)…"})
                    deadline = time.time() + max(60.0, 30.0 * len(all_synth))
                    while time.time() < deadline:
                        pending = [s for s in all_synth if vlm.get(s) is None]
                        if not pending:
                            break
                        time.sleep(0.5)
                    # final vote per track
                    for tid, sids in vlm_synth_ids.items():
                        if tid in vlm_db_updated:
                            continue
                        texts = []
                        for sid in sids:
                            r = vlm.get(sid)
                            if r:
                                texts.append(country.clean(r.raw))
                        if texts:
                            voted_text, _ = Counter(texts).most_common(1)[0]
                            self.store.upsert_plate(Plate(
                                session_id=session.id,
                                track_id=tid,
                                plate_text="",      # don't overwrite OCR
                                vlm_text=voted_text,
                                raw_text=" | ".join(texts),
                            ))
                            self.bus.publish({
                                "type": "vlm_verified",
                                "session_id": session.id,
                                "track_id": tid,
                                "vlm_text": voted_text,
                                "raw_text": " | ".join(texts),
                                "votes": texts,
                                "ts": time.time(),
                            })
                            vlm_db_updated.add(tid)

        except Exception as e:    # noqa: BLE001
            self.status.last_error = f"{type(e).__name__}: {e}"
            self.bus.publish({"type": "error", "message": self.status.last_error})
        finally:
            self.status.running = False
            self.bus.publish({"type": "session_ended",
                              "session_id": session.id})
            self.store.end_session(session.id)
