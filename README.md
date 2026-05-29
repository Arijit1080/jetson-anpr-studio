# Jetson ANPR Studio

**Real-time automatic license plate recognition on NVIDIA Jetson, with a full web UI.**

A self-hosted ANPR (Automatic Number Plate Recognition) platform that runs entirely on a Jetson Orin Nano (or any Jetson with JetPack 6.x). It takes input from a CSI ribbon camera, USB webcam, RTSP stream, image, or video file — detects license plates with a fine-tuned YOLO11 detector, reads them with your choice of three OCR engines, and optionally verifies the result with a vision–language model. Everything is exposed through a polished FastAPI + HTMX web app on port 8080.

Built originally as a YouTube tutorial project. Now Dockerized for one-command install on any Jetson.

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

# 2. Make sure your user can talk to docker without sudo.
#    (One-time. After this, log out and back in for it to take effect.)
sudo usermod -aG docker $USER

# 3. If a bare-metal Sparkler is already running, stop it so port 8080 is free.
sudo systemctl stop sparkler 2>/dev/null
sudo systemctl disable sparkler 2>/dev/null   # optional — prevents autostart

# 4. Authenticate to GHCR (only if the image is still private — see below).
echo <YOUR_GH_TOKEN> | docker login ghcr.io -u <YOUR_GH_USER> --password-stdin

# 5. Grab the compose file.
mkdir -p ~/jetson-anpr-studio && cd ~/jetson-anpr-studio
curl -O https://raw.githubusercontent.com/Arijit1080/jetson-anpr-studio/main/docker-compose.yml

# 6. Pull + start.
docker compose pull
docker compose up -d
docker compose logs -f          # watch first start (engine regen ~2 min)
```

First start regenerates TensorRT engines from the bundled `.pt` weights, which takes ~2 minutes per model. Subsequent restarts are instant.

---

## Pulling a private image

GHCR packages are **private by default** the first time GitHub Actions pushes them. The image has to be either made public (one-time, four clicks) or pulled with auth.

### Option 1 — Make the package public (simplest)

After the first successful CI build:

1. Open https://github.com/<your-user>/jetson-anpr-studio/pkgs/container/jetson-anpr-studio
2. Right sidebar → **Package settings** (gear icon)
3. Scroll to **Danger Zone** at the bottom → **Change visibility** → **Public**
4. Type the package name to confirm

After this, anyone — including a fresh Jetson with no GitHub login — can `docker pull` the image. This is the friction-free path if you want viewers / second machines to install with one command.

### Option 2 — Keep it private, log in with a token

Useful if you don't want the image public. Any GitHub Personal Access Token with the `read:packages` scope works. You can also reuse the OAuth token the `gh` CLI is using (`gh auth token`) — for your own packages it has implicit pull access, even without explicit `read:packages` scope.

```bash
# On the Jetson:
echo $TOKEN | docker login ghcr.io -u <your-gh-user> --password-stdin
docker pull ghcr.io/<your-gh-user>/jetson-anpr-studio:latest
```

The login token is cached in `~/.docker/config.json`, so you only do this once per machine.

> **Note**: if you're running `docker` with `sudo` (because your user isn't in the `docker` group yet), the auth cache is stored under `/root/.docker/`, not `~/.docker/`. Either run as the regular user (after `usermod` and re-login), or copy the config: `sudo cp -r ~/.docker /root/.docker`.

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
docker compose exec sparkler vi /app/sparklers_anpr/config/config.yaml
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

This builds the image locally and bind-mounts your `alpr/` and `sparklers_anpr/` directories. uvicorn runs with `--reload` so saving a `.py` file restarts the app immediately. No rebuild needed unless you change the Dockerfile or `requirements.txt`.

### Layout

```
jetson-anpr-studio/
├── alpr/                       # YOLO + OCR pipeline (importable module)
│   ├── core.py                 # ALPRPipeline class
│   ├── vlm_worker.py           # Florence-2 background worker
│   └── *.pt                    # YOLO11 plate-detector weights
├── sparklers_anpr/             # FastAPI web app
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

There are two supported workflows; choose based on what trade-off bothers you most.

### Recommended — Build natively on the Jetson, push to GHCR

This is how the current `:latest` was built. Native build means every `import torch` and `RUN python ...` step actually runs against the Jetson's real CUDA driver, so runtime mismatches surface during build (15-min cycles) instead of after a 50-min CI build + pull. Once the local image works, push it to GHCR as the shareable artifact.

```bash
# On the Jetson — clone (or git pull) the repo, then:
cd ~/jetson-anpr-studio
docker build -f docker/Dockerfile -t ghcr.io/arijit1080/jetson-anpr-studio:latest .
# 12-20 min on Orin Nano (full); ~3-5 min on subsequent code-only changes.

# Test locally first:
docker compose up -d
docker compose logs -f         # watch engine regen + verify
docker compose down

# Once happy, push to GHCR.
# First time on a machine: `docker login ghcr.io -u <user>` with a PAT
# that has write:packages scope.
docker push ghcr.io/arijit1080/jetson-anpr-studio:latest

# Also push a SHA-pinned tag for rollback safety:
SHA=$(git rev-parse --short HEAD)
docker tag  ghcr.io/arijit1080/jetson-anpr-studio:latest \
            ghcr.io/arijit1080/jetson-anpr-studio:sha-$SHA
docker push ghcr.io/arijit1080/jetson-anpr-studio:sha-$SHA
```

Disk note: the Orin Nano's 56 GB eMMC fills up quickly during builds. Free ≥ 15 GB before starting (`docker image prune -af` is your friend). The very large `dustynv/l4t-pytorch` base extracts to ~14 GB and won't fit — stick with `nvcr.io/nvidia/l4t-jetpack:r36.4.0` (~3 GB) which is what the current Dockerfile uses.

### Alternative — CI cross-build via GitHub Actions

Every push to `main` triggers `.github/workflows/build.yml` to cross-build the aarch64 image via QEMU and push it to GHCR. Useful for hands-off "ship from any commit" but slower (~25-30 min per build) and runtime issues only show up after a Jetson pull. Some library mismatches (CUDA driver version, native lib presence) only surface at runtime on a real Jetson, so the CI build can succeed while the resulting image fails to start. We use native build as the primary path for that reason.

Skip CI on a specific commit with `[skip ci]` in the commit subject.

### Alternative — Cross-build on a Mac

```bash
docker buildx create --use --name jetson-builder
docker buildx build \
    --platform linux/arm64 \
    -f docker/Dockerfile \
    -t ghcr.io/arijit1080/jetson-anpr-studio:latest \
    --push .
```

Slowest (~40 min on M-series Mac due to QEMU), same caveat as CI re: runtime-only mismatches.

---

## Why it's built this way

A few opinionated choices that came out of building this:

- **PyTorch from NVIDIA's Jetson wheel index**, not from PyPI. The PyPI wheels are CPU-only on aarch64; NVIDIA's wheel ships with CUDA 12.6 support.
- **paddlepaddle pinned to 2.6.2.** The 3.x line segfaults inside its PIR loader on aarch64 — confirmed reproducible on Orin Nano. The 2.x line is stable and 2.6.2 is the most recent.
- **paddleocr pinned to 2.7.3.** 2.10+ pulls in albumentations and python-docx, which transitively want to install `opencv-python` that clobbers the GStreamer-enabled system OpenCV. CSI camera support would silently break.
- **torch pinned to exactly 2.8.0 via direct wheel URL.** jetson-ai-lab's `/jp6/cu126/` index transitioned to cu130-built wheels starting with 2.9.1, even though the URL says cu126. Only 2.8.0 is genuinely a CUDA 12.6 build. *And* — pip's resolver picks PyPI's `2.8.0+cpu` over jetson-ai-lab's `2.8.0` because the `+cpu` local tag is considered "higher" version. We bypass the resolver entirely by installing from the exact wheel URL.
- **fast-plate-ocr model name uses the v2 ID.** 0.3.0 renamed `cct-xs-v1-global-model` → `global-plates-mobile-vit-v2-model`. `alpr/core.py` uses the new name; the older string just produces a "model not found" error at session start.
- **fast-plate-ocr 0.3+ requires grayscale input** `(H, W)` or `(H, W, 1)`. The previous `LicensePlateRecognizer` accepted BGR. `_ocr_fast` in core.py converts BGR → grayscale before calling `.run()` so both APIs work.
- **System OpenCV (apt python3-opencv), not pip.** The pip-installed wheels don't include GStreamer, which means nvarguscamerasrc — and therefore the IMX219 / IMX477 ribbon cameras — wouldn't work. The Dockerfile force-uninstalls any pip-installed `opencv-python` / `opencv-python-headless` after the requirements install so transitive deps can't sneak it in.
- **numpy pinned to 1.26.4.** The system cv2 was built against numpy 1.x ABI; numpy 2.x silently breaks `import cv2`.
- **`libopenblas0-pthread` installed via apt.** PyTorch on aarch64 links to `libopenblas.so.0` at load time; without it, `import torch` fails inside the container even though it works fine on a bare-metal Jetson (where openblas is pulled in as a transitive dep of other apt packages).
- **`onnx` + `onnxslim` in requirements.txt.** ultralytics YOLO → TensorRT engine export goes via ONNX as the intermediate format. Without these, `regen_engines.py` fails at first start with "No module named 'onnx'" and falls back to slow .pt inference.
- **`regen_engines.py` skips `.pt` files > 20 MB.** The YOLO11-L plate detector (50 MB) sends TensorRT into an OOM loop on the Orin Nano 8 GB — TRT keeps trying to allocate ~1 GB of working memory that doesn't fit. The size guard lets the small YOLO11-n model compile (~7 min, fits easily) while skipping L gracefully. L still loads from `.pt` at runtime; just slower per frame. Override via `REGEN_ENGINES_MAX_PT_MB=<n>` on larger Jetsons (Orin NX 16 GB, AGX Orin 64 GB).
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

### Freeing disk space on a tight Jetson

The image is ~6.7 GB and `docker pull` keeps a copy of every layer + the merged image, so on an Orin Nano eMMC (56 GB total) you want at least **10 GB free** before pulling. Common things to clean up:

```bash
# 1. User-level junk (safe, ~1-2 GB).
pip cache purge
rm -rf ~/.cache/huggingface ~/.paddlex ~/yolov8
sudo apt clean
sudo apt autoremove -y

# 2. If you've moved to Docker and don't need the bare-metal venv anymore
#    (~6.7 GB). Stop the systemd service first if it's still around.
sudo systemctl stop sparkler 2>/dev/null
sudo systemctl disable sparkler 2>/dev/null
rm -rf ~/yolo11/.venv

# 3. Recover swap space (~8 GB) if you had previously added two swapfiles
#    for Florence-2 GPU mode. Keep one + zram = 12 GB swap is plenty.
sudo swapoff /swapfile2 2>/dev/null
sudo rm -f /swapfile2
sudo sed -i.bak '/swapfile2/d' /etc/fstab
```

### Adding an M.2 NVMe (recommended for serious use)

The Orin Nano dev kit has an M.2 NVMe slot. Even a cheap 256 GB drive solves disk pressure permanently. After mounting it, relocate the docker data root:

```bash
sudo systemctl stop docker
sudo rsync -aHAX /var/lib/docker/ /mnt/nvme/docker/
sudo sed -i 's|"data-root":.*|"data-root": "/mnt/nvme/docker",|' /etc/docker/daemon.json 2>/dev/null \
    || echo '{"data-root": "/mnt/nvme/docker"}' | sudo tee /etc/docker/daemon.json
sudo systemctl start docker
```

---

## Troubleshooting

**`permission denied while trying to connect to the docker API at unix:///var/run/docker.sock`**
Your user isn't in the `docker` group yet. Either log out and back in after running `sudo usermod -aG docker $USER`, or for the current session use `sg docker -c 'docker compose up -d'` as a workaround. The `install.sh` script already handles this with the `sg docker` fallback.

**`Error response from daemon: Head ... : denied` / `401 Unauthorized` on `docker pull`**
The GHCR image is still private. Either flip the package to public on GitHub (see [Pulling a private image](#pulling-a-private-image)) or `docker login ghcr.io` with a token that has `read:packages` scope (or use `gh auth token`).

**Port 8080 already in use**
A bare-metal `sparkler.service` is still running (the systemd unit from the pre-Docker setup). Stop it with `sudo systemctl stop sparkler && sudo systemctl disable sparkler`. The `install.sh` script does this automatically when it detects the conflict.

**Docker login cached but `sudo docker pull` says 401**
`sudo docker` reads its config from `/root/.docker/`, not `~/.docker/`. Either run docker as your regular user (after the group fix above), or copy your auth: `sudo cp -r ~/.docker /root/.docker`.

**`Error: could not select device driver "nvidia" with capabilities`**
The nvidia-container-runtime isn't configured. Run `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.

**`docker info | grep nvidia` shows nothing on JetPack 6.x**
The nvidia-container-toolkit may have been removed or never installed. `sudo apt-get install -y nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.

**CSI camera doesn't appear in the source dropdown**
The `/tmp/argus_socket` mount or `privileged: true` flag was dropped. Check `docker compose config` and make sure both are present. Also confirm `nvargus-daemon.service` is running on the host (`systemctl is-active nvargus-daemon`).

**`numpy.core.multiarray failed to import`**
Some pip operation upgraded numpy past 2.x and broke the system cv2 4.8 ABI. The container's image has numpy pinned at 1.26.4 via `docker/constraints.txt`, so this should never happen inside it — but if you've been hacking inside the container with `--no-deps` or bind-mounted Python files, run `pip install numpy==1.26.4 --force-reinstall` inside the container.

**`The NVIDIA driver on your system is too old (found version 12060)` / `torch.cuda.is_available() == False`**
You're running an image built against CUDA 13 (torch ≥ 2.9) on a Jetson with CUDA 12.6 (JetPack 6.x). The Dockerfile pins `torch==2.8.0` specifically — that's the last release on jetson-ai-lab's `/jp6/cu126/` index that was actually built against cu126; everything 2.9.1 and newer there silently became cu130. Make sure you're pulling `:latest` from after that fix landed (`docker pull ghcr.io/.../jetson-anpr-studio:latest` then `docker compose up -d --force-recreate`).

**`ImportError: cannot import name 'LicensePlateRecognizer' from 'fast_plate_ocr'`**
fast-plate-ocr renamed the class to `ONNXPlateRecognizer` in 0.3.0. `alpr/core.py` already handles both names via a try/except import. If you hit this anyway, you're probably running an old image — pull the latest and recreate the container.

**Florence-2 OOM (`exit code 137` in compose logs)**
8 GB Orin Nano is tight. Switch to fast-plate-ocr or PaddleOCR alone (no VLM) from the dashboard. Or recreate the container so the GPU memory is fully freed: `docker compose down && docker compose up -d`. If you need VLM specifically and OOMs persist, restore the second 8 GB swapfile we dropped during disk cleanup (see the disk-management section of the project's commit history).

**First start hangs at "regenerating engines"**
Engine export takes ~60 s per model on Orin Nano. If it takes >5 min, check `docker compose logs` for errors. Often `--privileged` was missed and the GPU isn't accessible from inside the container.

**Apt complains `dpkg was interrupted, you must manually run 'sudo dpkg --configure -a'`**
A previous apt operation got killed mid-way (unrelated to Sparkler). Run `sudo dpkg --configure -a` to finish it. If a few packages are stuck on missing dependencies (e.g., `packagekit-tools` needing `packagekit`), remove them: `sudo apt-get remove --purge -y packagekit-tools gstreamer1.0-packagekit`.

**JetPack OTA half-applied across reboots**
If `cat /etc/nv_tegra_release` reports an older revision than expected, the OTA was queued but never fully applied. `sudo dpkg --configure -a && sudo apt autoremove -y && sudo reboot` will finish it. The Docker image targets `r36.4.0` which covers all R36.4.x patch releases.

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
