# Jetson ANPR Studio

**Real-time automatic license plate recognition on NVIDIA Jetson, with a full web UI.**

A self-hosted ANPR (Automatic Number Plate Recognition) platform that runs entirely on a Jetson Orin Nano (or any Jetson with JetPack 6.x). It takes input from a CSI ribbon camera, USB webcam, RTSP stream, image, or video file — detects license plates with a fine-tuned YOLO11 detector, reads them with your choice of three OCR engines, and optionally verifies the result with a vision–language model. Everything is exposed through a polished FastAPI + HTMX web app on port 8080.

Built originally as a YouTube tutorial project. Now Dockerized for one-command install on any Jetson.

![demo banner placeholder — drop a screenshot in docs/images/banner.png after first run](docs/images/banner.png)

---

## Features

- **Three OCR backends, switchable from the UI**
  - **fast-plate-ocr** (CCT-XS, ONNX) — ~6 ms/plate, plate-specific training, the default
  - **PaddleOCR PP-OCRv4** — ~30 ms/plate, general-purpose English OCR with strong digit recall
  - **EasyOCR** — ~80 ms/plate, fallback for unusual fonts
- **Optional VLM verification** — Florence-2 reads each plate independently; the UI marks a detection as **✓ verified** only when OCR and VLM agree
- **Multi-frame voting** — 5-frame majority vote on OCR text + 3 staggered VLM submissions per track to beat motion blur
- **ByteTrack tracking** — each car gets one stable row, not one per frame
- **Country profiles** — IN / UK / US / EU / Generic, controlling watermark stripping and character normalization
- **Live MJPEG preview** with overlay annotations
- **Server-sent events** stream every detection to all connected browsers in real time
- **Persistent log** — SQLite-backed with searchable history, per-plate detail page, CSV / JSON / crop-zip export
- **Multi-source input** — CSI (IMX219/IMX477), USB, RTSP/HTTP, image file, video file — all from the dashboard
- **One-command install** via Docker, with auto-pulled prebuilt images

---

## Hardware requirements

| Component | Tested with | Notes |
|---|---|---|
| Jetson | Orin Nano 8 GB (p3767-0005) | Works on Orin NX too. AGX should be even faster. |
| JetPack | 6.2 (L4T R36.4.x) | Required — the Docker image targets the R36.x ABI. |
| Camera | IMX219 CSI ribbon | USB webcams and RTSP also fully supported. |
| Storage | ≥ 10 GB free on `/` | The image is ~6.7 GB. Add an M.2 NVMe if your eMMC is tight. |
| Network | for first pull only | After install, no internet needed. |

---

## Quick start

### Option A — One-shot install (recommended)

On a fresh Jetson:

```bash
curl -fsSL https://raw.githubusercontent.com/Arijit1080/jetson-anpr-studio/main/install.sh | bash
```

This installs Docker + nvidia-container-toolkit if missing, pulls the prebuilt image from GHCR, and starts Sparkler. Open `http://<jetson-ip>:8080` in any browser on your network.

### Option B — Manual

```bash
# 1. Make sure docker + nvidia-container-toolkit are installed.
#    On JetPack 6.x they usually are; check with:
docker info | grep -i nvidia

# 2. Grab the compose file.
curl -O https://raw.githubusercontent.com/Arijit1080/jetson-anpr-studio/main/docker-compose.yml

# 3. Pull + start.
docker compose pull
docker compose up -d
```

First start regenerates TensorRT engines from the bundled `.pt` weights, which takes ~2 minutes per model. Subsequent restarts are instant.

---

## Using the UI

Open `http://<jetson-ip>:8080`. The dashboard has:

- **Source** — pick CSI / USB / RTSP / image file / video file
- **Detector** — YOLO11-n plate (fastest) or YOLO11-L (most accurate)
- **OCR backend** — fast-plate-ocr / PaddleOCR / EasyOCR / + Florence-2 VLM
- **Country** — IN / UK / US / EU / generic (changes plate cleaning rules)

Hit **Start**. The live preview comes up on the right. Every detection appears in the right-side log instantly, and the full log is at `/log`. Click any plate text to see all sightings of that plate.

---

## Architecture

```
┌──────────────┐   ┌─────────────────┐   ┌──────────────────┐
│   Source     │──▶│ YOLO11 plate    │──▶│ ByteTrack        │
│ CSI/USB/RTSP │   │ detector (TRT)  │   │ tracker          │
│ image/video  │   └─────────────────┘   └──────────────────┘
└──────────────┘                                  │
                                                  ▼
                          ┌──────────────────────────────────────┐
                          │  per-track crop                       │
                          └──────────────────────────────────────┘
                                       │
                ┌──────────────────────┼──────────────────────┐
                ▼                      ▼                      ▼
        ┌───────────────┐    ┌───────────────┐      ┌────────────────┐
        │ fast-plate    │    │ PaddleOCR     │      │ EasyOCR        │
        │ -ocr (ONNX)   │    │ PP-OCRv4      │      │                │
        └───────────────┘    └───────────────┘      └────────────────┘
                │                      │                      │
                └──────────┬───────────┴──────────────────────┘
                           ▼
                  ┌────────────────┐      ┌───────────────────┐
                  │ 5-frame vote   │      │ Florence-2 VLM    │
                  │ (Counter)      │      │ (optional)        │
                  └────────────────┘      └───────────────────┘
                           │                       │
                           ▼                       ▼
                  ┌──────────────────────────────────────┐
                  │  SQLite store + crop archive          │
                  └──────────────────────────────────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │ FastAPI web UI  │
                          │ MJPEG + SSE     │
                          └─────────────────┘
```

---

## Configuration

`config.yaml` is generated on first start and lives in the `sparkler_config` Docker volume. You can edit it directly:

```bash
docker compose exec sparkler vi /app/sparkler/config/config.yaml
docker compose restart
```

Or change settings from the **Settings** page in the UI.

---

## Development

You want hot-reload — edit Python on your laptop, see changes in 2 seconds:

```bash
git clone https://github.com/Arijit1080/jetson-anpr-studio.git
cd jetson-anpr-studio
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

This builds the image locally and bind-mounts your `alpr/` and `sparkler/` directories. uvicorn runs with `--reload` so saving a `.py` file restarts the app immediately. No rebuild needed unless you change the Dockerfile or `requirements.txt`.

### Layout

```
jetson-anpr-studio/
├── alpr/                       # YOLO + OCR pipeline (importable module)
│   ├── core.py                 # ALPRPipeline class
│   ├── vlm_worker.py           # Florence-2 background worker
│   └── *.pt                    # YOLO11 plate-detector weights
├── sparkler/                   # FastAPI web app
│   ├── app.py                  # routes + endpoints
│   ├── pipeline.py             # PipelineRunner (background thread)
│   ├── config.py               # config schema + load/save
│   ├── countries.py            # plate-cleaning profiles
│   ├── db.py                   # SQLite store
│   ├── sources.py              # CSI / USB / RTSP / file source adapters
│   ├── static/                 # CSS, JS
│   └── templates/              # Jinja2 HTML (HTMX + Alpine.js)
├── docker/
│   ├── Dockerfile              # the build recipe
│   ├── entrypoint.sh           # regen TRT engines → exec uvicorn
│   ├── regen_engines.py        # rebuild .engine from .pt on first start
│   ├── prefetch_models.py      # bundle Florence-2 / paddle / fast-plate at build
│   ├── requirements.txt        # main pip deps
│   └── requirements-paddle.txt # paddle deps (installed --no-deps)
├── .github/workflows/build.yml # CI: cross-build aarch64 → push to GHCR
├── docker-compose.yml          # production (image: ghcr.io/...)
├── docker-compose.dev.yml      # dev overlay (build: + bind-mounts)
└── install.sh                  # one-shot Jetson installer
```

---

## Building the image yourself

Every push to `main` triggers GitHub Actions to cross-build the aarch64 image and push it to `ghcr.io/arijit1080/jetson-anpr-studio:latest`. You can also build locally:

### On a Mac (cross-compile via QEMU)

```bash
docker buildx create --use --name jetson-builder
docker buildx build \
    --platform linux/arm64 \
    -f docker/Dockerfile \
    -t jetson-anpr-studio:local \
    --load .
```

Takes ~40 min on an M-series Mac. Result lives in your local Docker. You can `docker save` it to a tar and `scp` to a Jetson.

### On the Jetson natively

```bash
docker build -f docker/Dockerfile -t jetson-anpr-studio:local .
```

Faster (~25 min, no QEMU) but needs ~15 GB free disk.

---

## Why it's built this way

A few opinionated choices that came out of building this:

- **PyTorch from NVIDIA's Jetson wheel index**, not from PyPI. The PyPI wheels are CPU-only on aarch64; NVIDIA's wheel ships with CUDA 12.6 support.
- **paddlepaddle pinned to 2.6.2.** The 3.x line segfaults inside its PIR loader on aarch64 — confirmed reproducible on Orin Nano. The 2.x line is stable and 2.6.2 is the most recent.
- **paddleocr pinned to 2.7.3.** 2.10+ pulls in albumentations and python-docx, which transitively want to install `opencv-python` that clobbers the GStreamer-enabled system OpenCV. CSI camera support would silently break.
- **System OpenCV (apt python3-opencv), not pip.** The pip-installed wheels don't include GStreamer, which means nvarguscamerasrc — and therefore the IMX219 / IMX477 ribbon cameras — wouldn't work.
- **numpy pinned to 1.26.4.** The system cv2 4.8 was built against numpy 1.x ABI; numpy 2.x silently breaks `import cv2`.
- **TensorRT engines regenerated on first start.** Engines aren't portable across TRT minor versions, so bundling them in the image would break on any JetPack patch upgrade. The `.pt` weights are bundled instead.
- **Multi-frame voting + VLM cross-check.** Single-frame OCR is fragile (motion blur, occlusion). Voting across 5 frames and cross-checking with Florence-2 catches most errors without needing a perfect detector.

---

## Operations

### Update to a new version

```bash
docker compose pull
docker compose up -d
```

Your SQLite database, crops, and config survive (they're in named volumes, not the image).

### Roll back

```bash
docker compose down
docker compose pull ghcr.io/arijit1080/jetson-anpr-studio:sha-abc1234
docker compose up -d
```

Every CI build tags both `:latest` and `:sha-<short>`, so any past commit is a one-line rollback.

### Backup the database + crops

```bash
docker run --rm \
    -v sparkler_runs:/data \
    -v "$(pwd)":/backup \
    alpine \
    tar czf /backup/sparkler-backup-$(date +%F).tar.gz -C /data .
```

### Logs

```bash
docker compose logs -f                 # tail live
docker compose logs --since 1h         # last hour
```

### Shell inside the container

```bash
docker compose exec sparkler bash
```

---

## Troubleshooting

**`Error: could not select device driver "nvidia" with capabilities`**
The nvidia-container-runtime isn't configured. Run `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.

**CSI camera doesn't appear in the source dropdown**
The `/tmp/argus_socket` mount or `privileged: true` flag was dropped. Check `docker compose config` and make sure both are present. Also confirm `nvargus-daemon.service` is running on the host.

**`numpy.core.multiarray failed to import`**
Some pip operation upgraded numpy past 2.x. The container's image has numpy pinned, so this should never happen inside the container — but if you've been hacking with `--no-deps` or bind-mounted Python files, run `pip install numpy==1.26.4 --force-reinstall` inside the container.

**Florence-2 OOM**
8 GB Orin Nano is tight. Switch to fast-plate-ocr alone (no VLM) from the dashboard. Or restart the container so the GPU memory is fully freed before re-attempting.

**First start hangs at "regenerating engines"**
Engine export takes ~60 s per model on Orin Nano. If it takes >5 min, check `docker compose logs` for errors. Often `--privileged` was missed and the GPU isn't accessible.

---

## License

[MIT](LICENSE).

---

## Credits

Built on top of a lot of great open source:
- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Florence-2](https://huggingface.co/microsoft/Florence-2-base) by Microsoft
- [FastAPI](https://fastapi.tiangolo.com/), [HTMX](https://htmx.org/), [Alpine.js](https://alpinejs.dev/), [Tailwind CSS](https://tailwindcss.com/)

Plate-detector weights fine-tuned on the [Roboflow Universe license-plate dataset](https://universe.roboflow.com/).
