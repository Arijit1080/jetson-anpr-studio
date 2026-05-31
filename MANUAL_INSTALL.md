# Manual install on a fresh Jetson (no Docker)

This guide walks you through installing **Jetson ANPR Studio** directly on the host — no containers — on a fresh NVIDIA Jetson with JetPack 6.x. Every command is copy-paste ready, every version pin is annotated with the reason it exists, and the troubleshooting section at the bottom calls out the gotchas we hit while building this.

If you'd rather skip all of this and run the pre-built Docker image, see the **Quick start** section of the [README](./README.md) — one command and you're up.

---

## Why you might want the manual install (vs Docker)

| | **Docker** | **Manual / bare-metal** |
|---|---|---|
| First-time setup | 2 commands, ~5 min image pull | ~45 min of pip installs |
| Update to new code | `docker compose pull` | `git pull && systemctl restart` |
| Disk used | ~13 GB (image) | ~7 GB (venv + system) |
| GPU memory headroom | Slight container overhead | A few hundred MB less overhead |
| Customise the code in-place | Bind-mount in dev profile | Edit + restart, no rebuild |
| Reproducibility across Jetsons | Identical to the byte | Depends on what's currently on PyPI |

For most users — **Docker is the right choice.** This guide exists for: people who want to learn how everything fits together, people who want to modify the code without rebuilding images, and people on extremely tight disk where the Docker image won't fit.

---

## What you'll need

| | |
|---|---|
| Jetson | Orin Nano 8 GB (tested), Orin NX, AGX Orin |
| JetPack | 6.x (L4T R36.4.x) — verify with `cat /etc/nv_tegra_release` |
| Free disk | At least 8 GB on `/` |
| Network | The pip installs pull ~3 GB of wheels |
| Time | ~45 min total, mostly waiting on pip + a one-time TRT engine compile |

If you've never used a Jetson before — flash JetPack 6.2 from NVIDIA SDK Manager first, get to a desktop / SSH, then come back here.

---

## Step 1 — System packages (apt)

```bash
sudo apt update
sudo apt install -y \
    python3 python3-pip python3-dev python3-venv python3-setuptools python3-wheel \
    python3-opencv \
    gstreamer1.0-tools gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav \
    libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg \
    libopenblas0-pthread \
    build-essential pkg-config git curl
```

**Why each non-obvious one:**
- `python3-opencv` — the **system OpenCV with GStreamer support**, needed for CSI camera (`nvarguscamerasrc`). The pip-installed `opencv-python` wheels don't include GStreamer and would silently break the IMX219 / IMX477 ribbon cameras.
- `libopenblas0-pthread` — PyTorch on aarch64 links to `libopenblas.so.0` at load time. Without it, `import torch` fails with `ImportError: libopenblas.so.0: cannot open shared object file`.
- `gstreamer1.0-plugins-*` — the camera + video decoders feeding `cv2.VideoCapture`.

---

## Step 2 — Optional but recommended: increase swap

The 8 GB Orin Nano is genuinely tight when running Florence-2 VLM. An extra 8 GB swapfile gives the kernel breathing room.

```bash
# Skip if you already have ≥ 12 GB of swap (check with: swapon --show)
sudo fallocate -l 8G /swapfile2
sudo chmod 600 /swapfile2
sudo mkswap /swapfile2
sudo swapon /swapfile2
echo '/swapfile2 none swap sw 0 0' | sudo tee -a /etc/fstab
swapon --show
```

(JetPack ships with 1 swapfile and zram. We're adding a second.)

---

## Step 3 — Clone the repo

```bash
git clone https://github.com/Arijit1080/jetson-anpr-studio.git ~/jetson-anpr-studio
cd ~/jetson-anpr-studio
```

The `alpr/` directory has YOLO + OCR pipeline code AND the bundled YOLO11 plate-detection weights. The `sparklers_anpr/` directory has the FastAPI web app. The `docker/` directory has the requirements files — we'll reuse those.

---

## Step 4 — Create the Python virtualenv

```bash
# Put it under ~/yolo11/.venv so the included systemd unit (Step 9) finds it.
mkdir -p ~/yolo11
python3 -m venv ~/yolo11/.venv
source ~/yolo11/.venv/bin/activate
python --version    # should be 3.10.x
pip install --upgrade pip
```

Every subsequent `pip install` happens **inside this activated venv**. If you open a new shell, re-run `source ~/yolo11/.venv/bin/activate`.

---

## Step 5 — Install PyTorch (CUDA 12.6, the only version that works)

This is THE critical step. JetPack 6.2's CUDA driver is 12.6. Most PyTorch wheels you'll find online were built against CUDA 13 and will silently load but `torch.cuda.is_available()` returns `False`.

The only working wheel on jetson-ai-lab.io's `/jp6/cu126/` slot is **torch 2.8.0**. Install via direct URL to bypass pip's resolver (which otherwise prefers PyPI's `2.8.0+cpu` wheel):

```bash
pip install 'numpy==1.26.4'
pip install \
    'https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl' \
    'https://pypi.jetson-ai-lab.io/jp6/cu126/+f/907/c4c1933789645/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl'
```

Verify it actually works:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# Expected:  2.8.0 12.6 True
```

If you see `False` or a CUDA-version mismatch error, **stop here**. You've either downloaded the wrong wheel or your JetPack isn't 6.2. See the Troubleshooting section.

**Why `numpy==1.26.4`:** the apt-installed `python3-opencv` (4.5.4) was built against numpy 1.x's ABI. Any numpy 2.x breaks `import cv2` with `numpy.core.multiarray failed to import`.

---

## Step 6 — Install the main Python requirements

```bash
pip install -r ~/jetson-anpr-studio/docker/requirements.txt
```

This installs FastAPI, ultralytics, transformers, easyocr, fast-plate-ocr, onnx + onnxslim (for TensorRT engine export), timm (for Florence-2), and friends. Takes ~15-20 min on a typical home connection.

---

## Step 7 — Install PaddleOCR (with the `--no-deps` trick)

Paddle has a few transitive dependencies that would *clobber* the system OpenCV. We install with `--no-deps` and explicitly list the deps that don't conflict:

```bash
pip install --no-deps -r ~/jetson-anpr-studio/docker/requirements-paddle.txt
```

Force numpy back to 1.26.4 in case a transitive dep bumped it:

```bash
pip install --force-reinstall --no-deps 'numpy==1.26.4'
```

Remove any pip-installed OpenCV that snuck in (so the system cv2 with GStreamer remains the one Python imports):

```bash
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python || true
```

---

## Step 8 — Symlink the system cv2 into the venv

Python venvs are isolated from system packages by default, so the apt-installed `cv2.cpython-310-aarch64-linux-gnu.so` isn't visible from inside the venv. Symlink it:

```bash
ln -sfn /usr/lib/python3/dist-packages/cv2.cpython-310-aarch64-linux-gnu.so \
        ~/yolo11/.venv/lib/python3.10/site-packages/cv2.cpython-310-aarch64-linux-gnu.so
```

Verify (must show `GStreamer: YES`):

```bash
python -c "import cv2; print(cv2.__version__); print('GStreamer:', 'YES' if 'GStreamer' in cv2.getBuildInformation() else 'NO')"
# Expected:  4.5.4 \n GStreamer: YES
```

---

## Step 9 — First run

The pipeline finds the alpr code via `PYTHONPATH`. Run:

```bash
cd ~/jetson-anpr-studio/sparklers_anpr
ALPR_DIR=$HOME/jetson-anpr-studio/alpr \
PYTHONPATH=$HOME/jetson-anpr-studio/alpr:$HOME/jetson-anpr-studio/sparklers_anpr \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6 \
PYTORCH_NVML_BASED_CUDA_CHECK=0 \
~/yolo11/.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8080
```

Open `http://<your-jetson-ip>:8080/` in a browser on the same network. You should see the dashboard.

**First time you hit "Start"** with a plate image, the YOLO11-n TensorRT engine compiles in the background (~7 min on Orin Nano). Subsequent runs are instant — the `.engine` file is cached at `~/jetson-anpr-studio/alpr/license-plate-finetune-v1n.engine`.

---

## Step 10 (optional) — Install as a systemd service so it autostarts on boot

```bash
sudo tee /etc/systemd/system/sparkler.service > /dev/null <<EOF
[Unit]
Description=Sparkler ANPR Studio (FastAPI web UI for plate recognition)
After=network-online.target nvargus-daemon.service
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/jetson-anpr-studio/sparklers_anpr
Environment=PYTHONUNBUFFERED=1
Environment=ALPR_DIR=$HOME/jetson-anpr-studio/alpr
Environment=PYTHONPATH=$HOME/jetson-anpr-studio/alpr:$HOME/jetson-anpr-studio/sparklers_anpr
Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6
Environment=PYTORCH_NVML_BASED_CUDA_CHECK=0
ExecStart=$HOME/yolo11/.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8080
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now sparkler
sudo systemctl status sparkler
```

After this, Sparkler comes up automatically every time the Jetson boots. Stop/start manually with `sudo systemctl stop sparkler` / `sudo systemctl start sparkler`.

---

## Updating later

```bash
cd ~/jetson-anpr-studio
git pull
sudo systemctl restart sparkler    # (or kill the uvicorn process if not using systemd)
```

If `git pull` brings in changes to `docker/requirements.txt`, re-run the relevant `pip install` from Step 6 / 7.

---

## Troubleshooting

**`torch.cuda.is_available()` returns False**
You have the wrong PyTorch wheel. Re-do Step 5 — install **exactly** `torch==2.8.0` from the jetson-ai-lab `/jp6/cu126/+f/...` URL above. The `pypi.jetson-ai-lab.io/jp6/cu126/+simple/` index serves cu130 wheels for everything 2.9.1+; only 2.8.0 is real cu126.

**`numpy.core.multiarray failed to import` when importing cv2**
numpy got bumped past 2.x. Re-pin: `pip install --force-reinstall --no-deps 'numpy==1.26.4'`.

**`ImportError: libopenblas.so.0: cannot open shared object file`**
Step 1 missed `libopenblas0-pthread`. `sudo apt install -y libopenblas0-pthread` fixes it.

**`ModuleNotFoundError: No module named 'cv2'` from inside the venv**
The symlink from Step 8 is missing. Re-create it.

**`from paddleocr import PaddleOCR` raises `ModuleNotFoundError: No module named 'opt_einsum'`**
paddlepaddle 2.6.2 imports `opt_einsum` unconditionally. We pin it in `requirements-paddle.txt`; if it's missing: `pip install opt-einsum==3.3.0`.

**Florence-2 VLM dies with `ModuleNotFoundError: No module named 'timm'`**
`pip install 'timm>=0.9'`. (It's in `requirements.txt` already; only happens if pip's transitive resolve skipped it for some reason.)

**Florence-2 VLM dies with `'NoneType' object has no attribute 'shape'` in `prepare_inputs_for_generation`**
That's a transformers ≥ 4.46 incompatibility with Florence-2's community-maintained modeling file. Our `alpr/vlm_worker.py` passes `use_cache=False` to bypass it. If you've edited that file, restore the kwarg.

**Florence-2 VLM dies with `AttributeError: 'Florence2ForConditionalGeneration' object has no attribute '_supports_sdpa'`**
Same kind of incompatibility. Our `vlm_worker.py` monkey-patches `PreTrainedModel._supports_sdpa = False` before loading. If you've edited that file, restore the monkey-patch.

**First VLM GPU verification after process start returns empty `vlm_text`**
PyTorch's CUDA caching allocator hits an `NVML_SUCCESS == r INTERNAL ASSERT FAILED` on the very first allocation on Tegra. The worker catches and skips. Every subsequent plate works fine (~1.3 s). Use `fast+vlm-cpu` if you can't afford to skip the first plate.

**Engine compile hangs spitting `NvMapMemAllocInternalTagged ... error 12` for the L model**
YOLO11-L is too big for TensorRT to compile on the 8 GB Orin Nano. Our `regen_engines.py` skips weights > 20 MB by default. The L model still loads from `.pt` at runtime (slower per frame, but works).

**Port 8080 already in use**
Something else is on it. Either stop that thing (`sudo systemctl stop sparkler` if the systemd unit is running, or `docker compose down` if a previous Docker install is up) or change the port in the uvicorn command (and in the systemd unit if you set that up).

**`E: dpkg was interrupted` from apt**
Some earlier apt operation crashed mid-way. Run `sudo dpkg --configure -a` to finish it. If a few packages are stuck on missing dependencies, remove the orphans: `sudo apt-get remove --purge -y packagekit-tools gstreamer1.0-packagekit` (or whichever ones apt names).

---

## What's in the repo, briefly

```
jetson-anpr-studio/
├── alpr/                        ← YOLO + OCR pipeline (the brain)
│   ├── core.py                  ← ALPRPipeline class
│   ├── vlm_worker.py            ← Florence-2 worker thread
│   └── *.pt                     ← YOLO11 plate-detector weights
├── sparklers_anpr/              ← FastAPI web app (what you see in the browser)
│   ├── app.py                   ← routes / endpoints
│   ├── pipeline.py              ← PipelineRunner background thread
│   ├── config.py                ← config schema + load/save
│   └── templates/               ← HTML for the dashboard, log, settings pages
├── docker/                      ← Dockerfile + requirements (we reuse the
│                                   requirements files for the manual install)
└── README.md                    ← Docker / one-command install path
```

The full architecture diagram is in the [README](./README.md#architecture).

---

## Next steps

- Open the dashboard, drop in a plate image, hit Start, see what each OCR backend reads.
- Switch to a CSI / USB camera and watch live MJPEG.
- Browse the source — `alpr/core.py` and `sparklers_anpr/pipeline.py` are the entire engine, ~600 lines together.
