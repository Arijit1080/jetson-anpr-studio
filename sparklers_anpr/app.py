"""Sparkler ANPR Studio — FastAPI app.

Run:
    cd ~/sparklers_anpr
    source ~/yolo11/.venv/bin/activate
    uvicorn app:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import shutil
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                                PlainTextResponse, RedirectResponse,
                                Response, StreamingResponse)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import config as cfg_mod
from db import Store
from pipeline import EventBus, PipelineRunner
from sources import SourceSpec, list_usb_cameras


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
UPLOADS = RUNS / "uploads"
CROPS = RUNS / "crops"
RUNS.mkdir(exist_ok=True)
UPLOADS.mkdir(exist_ok=True)
CROPS.mkdir(exist_ok=True)

# Global runtime state
config = cfg_mod.load()
store = Store(RUNS / "sparkler.db")
bus = EventBus()
runner = PipelineRunner(store=store, bus=bus, runs_dir=RUNS)

app = FastAPI(title=config.ui.title)
templates = Jinja2Templates(directory=str(HERE / "templates"))

# Custom Jinja2 filters
def _fmt_time(ts) -> str:
    """Format a unix timestamp as YYYY-MM-DD HH:MM:SS in local time."""
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return str(ts)


def _fmt_time_short(ts) -> str:
    """Just HH:MM:SS."""
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S")
    except (ValueError, TypeError):
        return str(ts)


templates.env.filters["fmt_time"] = _fmt_time
templates.env.filters["fmt_time_short"] = _fmt_time_short
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")
app.mount("/crops", StaticFiles(directory=str(CROPS)), name="crops")


def _warmup_models():
    """Preload the configured detector + OCR backend (and VLM if it's part
    of the default backend) so the first /api/start doesn't pay the 5–10s
    cold-load cost.  Runs in a background thread from the startup hook.

    Cached models live on `runner._cached_pipeline` / `runner._cached_vlm`
    and stay resident until the FastAPI shutdown hook.  Subsequent sessions
    reuse them via `pipeline._build_pipeline()`'s cache check."""
    print(f"[warmup] preloading for ocr_backend={config.pipeline.ocr_backend!r} …",
          flush=True)
    t0 = time.time()
    try:
        runner._build_pipeline(config.pipeline)
        # One dummy detect so TRT JIT-compiles kernels + allocates streams.
        import numpy as np
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        try:
            runner._pipeline.detector.predict(
                dummy, imgsz=config.pipeline.imgsz, device=0,
                half=runner._pipeline._is_engine, verbose=False,
            )
        except Exception as e:    # noqa: BLE001
            print(f"[warmup] dummy detect failed (non-fatal): {e}", flush=True)
        # Per-session pointers go back to None; cached singletons stay warm.
        runner._pipeline = None
        runner._vlm = None
        print(f"[warmup] ready in {time.time() - t0:.1f}s", flush=True)
    except Exception as e:    # noqa: BLE001
        print(f"[warmup] failed (non-fatal): {e}", flush=True)


@app.on_event("startup")
async def _on_startup():
    bus.attach_loop(asyncio.get_event_loop())
    # Kick the warmup off as a background task so FastAPI starts answering
    # /healthz etc. immediately.  The first /api/start request will block
    # waiting on the cache only if warmup hasn't finished yet — usually
    # warmup is done in 10–15s, by which time the user hasn't even loaded
    # the dashboard.
    asyncio.get_event_loop().run_in_executor(None, _warmup_models)


@app.on_event("shutdown")
async def _on_shutdown():
    runner.stop()
    # Release cached singletons (VLM thread + GPU memory).  Without this,
    # the process can hang on exit waiting for the VLM thread.
    runner.shutdown_models()


# ---------- helpers --------------------------------------------------------

def _page_ctx(request: Request, **extra):
    return {"request": request, "cfg": config, "extra": extra}


# ---------- pages ----------------------------------------------------------
# NOTE: starlette 1.x changed TemplateResponse — request is the FIRST arg.
# Keyword form is the most portable across versions.

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(
        request=request, name="dashboard.html",
        context={"cfg": config, "status": runner.status,
                 "usb_cameras": list_usb_cameras()},
    )


@app.get("/log", response_class=HTMLResponse)
async def log_page(request: Request, q: str = "", valid: int = 0):
    rows = store.recent(limit=200, search=q, verified_only=bool(valid))
    return templates.TemplateResponse(
        request=request, name="log.html",
        context={"cfg": config, "rows": rows, "q": q,
                 "valid_only": bool(valid)},
    )


@app.get("/plate/{text}", response_class=HTMLResponse)
async def plate_page(request: Request, text: str):
    rows = store.by_text(text)
    if not rows:
        raise HTTPException(404)
    return templates.TemplateResponse(
        request=request, name="plate.html",
        context={"cfg": config, "text": text.upper(), "rows": rows},
    )


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    return templates.TemplateResponse(
        request=request, name="settings.html",
        context={"cfg": config},
    )


# ---------- control API ----------------------------------------------------

@app.post("/api/start")
async def api_start(
    type: str = Form(...),
    uri: str = Form(""),
    csi_sensor_id: int = Form(0),
    flip_method: int = Form(2),
    backend: str = Form(...),
    detector: str = Form(...),
    det_conf: float = Form(0.20),
    imgsz: int = Form(640),
):
    config.source.type = type
    config.source.uri = uri
    config.source.csi_sensor_id = csi_sensor_id
    config.source.flip_method = flip_method
    config.pipeline.ocr_backend = backend
    config.pipeline.detector_path = detector
    config.pipeline.detector_conf = det_conf
    config.pipeline.imgsz = imgsz
    cfg_mod.save(config)

    spec = SourceSpec(
        type=type, uri=uri,
        csi_sensor_id=csi_sensor_id, flip_method=flip_method,
        width=config.source.width, height=config.source.height,
        fps=config.source.fps,
    )
    runner.start(spec, config.pipeline)
    return {"ok": True, "status": runner.status.__dict__}


@app.post("/api/stop")
async def api_stop():
    runner.stop()
    return {"ok": True}


@app.post("/api/settings")
async def api_settings(
    theme: str = Form("dark"),
    play_sound: int = Form(1),
    keep_log_days: int = Form(30),
    crops_max_gb: float = Form(4.0),
    vlm_model: str = Form("microsoft/Florence-2-base"),
    vote_frames: int = Form(5),
    country: str = Form("IN"),
):
    config.ui.theme = theme
    config.ui.play_sound_on_detect = bool(play_sound)
    config.ui.keep_log_days = keep_log_days
    config.ui.crops_max_gb = crops_max_gb
    config.pipeline.vlm_model = vlm_model
    config.pipeline.vote_frames = vote_frames
    config.pipeline.country = country.upper()
    cfg_mod.save(config)
    return RedirectResponse("/settings", status_code=303)


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "upload.bin").suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp",
                   ".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        raise HTTPException(400, f"unsupported file type {ext}")
    dest = UPLOADS / f"{int(time.time()*1000)}{ext}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    is_video = ext in {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    return {"ok": True, "path": str(dest),
            "type": "video" if is_video else "image"}


# ---------- streams --------------------------------------------------------

@app.get("/stream.mjpg")
async def mjpg_stream():
    boundary = "frame"
    async def gen():
        # send something even when paused
        blank = bytes([255, 216, 255, 224]) + b"\x00\x10JFIF" + b"\x00" * 12
        last_sent: Optional[bytes] = None
        while True:
            jpg = runner.latest_jpg() or last_sent
            if jpg is None:
                # placeholder ~1 KB grey frame
                import numpy as np, cv2 as _cv2
                grey = np.full((360, 640, 3), 32, dtype=np.uint8)
                _cv2.putText(grey, "no source running", (40, 200),
                             _cv2.FONT_HERSHEY_SIMPLEX, 1, (180,180,180), 2)
                ok, buf = _cv2.imencode(".jpg", grey)
                jpg = buf.tobytes() if ok else blank
            last_sent = jpg
            yield (b"--" + boundary.encode() + b"\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                   + jpg + b"\r\n")
            await asyncio.sleep(1/25.0)
    return StreamingResponse(
        gen(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
    )


@app.get("/events")
async def sse_events():
    q = bus.subscribe()
    async def gen():
        try:
            while True:
                evt = await q.get()
                yield f"data: {json.dumps(evt)}\n\n"
        finally:
            bus.unsubscribe(q)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ---------- deletion -------------------------------------------------------

@app.post("/api/clear/all")
async def api_clear_all(also_crops: int = Form(0)):
    """Delete every plate row.  If also_crops=1, wipe the crop image dir too."""
    n = store.delete_all()
    if also_crops:
        for child in CROPS.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    child.unlink()
                except OSError:
                    pass
    return RedirectResponse("/log", status_code=303)


@app.post("/api/clear/{plate_id}")
async def api_clear_one(plate_id: int):
    row = store.get(plate_id)
    if row and row.get("crop_path"):
        try:
            (CROPS / row["crop_path"]).unlink()
        except OSError:
            pass
    ok = store.delete_by_id(plate_id)
    return {"ok": bool(ok), "deleted": plate_id}


@app.post("/api/clear/text/{text}")
async def api_clear_text(text: str):
    n = store.delete_by_text(text)
    return {"ok": True, "deleted": n, "text": text}


# ---------- status / stats -------------------------------------------------

@app.get("/api/status")
async def api_status():
    return {"runner": runner.status.__dict__, "stats": store.stats()}


@app.get("/api/recent")
async def api_recent(limit: int = 20, q: str = "", valid: int = 0):
    return store.recent(limit=limit, search=q, verified_only=bool(valid))


# ---------- export ---------------------------------------------------------

@app.get("/export/csv")
async def export_csv():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["plate_text", "n_frames", "valid", "vlm_verified",
                "first_seen", "last_seen", "session_id"])
    for r in store.recent(limit=10_000):
        w.writerow([r["plate_text"], r["n_frames"], r["valid"],
                    r["vlm_verified"], r["first_seen"], r["last_seen"],
                    r["session_id"]])
    return PlainTextResponse(buf.getvalue(),
                             headers={"Content-Disposition":
                                      "attachment; filename=plates.csv"})


@app.get("/export/json")
async def export_json():
    return JSONResponse(store.recent(limit=10_000),
                        headers={"Content-Disposition":
                                 "attachment; filename=plates.json"})


@app.get("/export/crops.zip")
async def export_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in store.recent(limit=10_000):
            if not r["crop_path"]:
                continue
            src = CROPS / r["crop_path"]
            if src.exists():
                arc = f"{r['plate_text'] or 'unknown'}/{src.name}"
                zf.write(src, arcname=arc)
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition":
                                      "attachment; filename=plate_crops.zip"})
