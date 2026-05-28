# syntax=docker/dockerfile:1.7
#
# Jetson ANPR Studio — Dockerfile
# ---------------------------------
# Targets the NVIDIA Jetson Orin / Xavier line on JetPack 6.x (L4T R36.x).
# Layers are ordered slowest-to-change → fastest-to-change so that editing
# Python code rebuilds in seconds, not minutes.
#
# Build with:
#   docker buildx build --platform linux/arm64 -t jetson-anpr-studio .
# Or on the Jetson itself:
#   docker build -t jetson-anpr-studio .
#
# Run with: see docker-compose.yml

ARG L4T_TAG=r36.4.0
FROM nvcr.io/nvidia/l4t-jetpack:${L4T_TAG}

# ---------------------------------------------------------------------------
# Stage 0 — system packages (apt)
# ---------------------------------------------------------------------------
# We need:
#   - python3 (3.10 on this base)
#   - python3-opencv: the SYSTEM OpenCV 4.x with GStreamer support.  This is
#     critical: pip-installed opencv-python doesn't have GStreamer, so CSI
#     camera input wouldn't work.
#   - GStreamer plugins (good/bad/ugly) + nvidia's nvarguscamerasrc helpers
#     so the IMX219 / IMX477 ribbon cameras feed through cleanly.
#   - libgl1 / libglib2.0-0 are runtime deps of OpenCV.
#   - tzdata for sane timestamp printing.
# ---------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev python3-setuptools python3-wheel \
        python3-opencv \
        gstreamer1.0-tools gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly gstreamer1.0-libav \
        libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        ffmpeg \
        tzdata ca-certificates curl \
        build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Stage 1 — PyTorch (NVIDIA's Jetson wheel)
# ---------------------------------------------------------------------------
# NVIDIA publishes prebuilt PyTorch wheels matching each JetPack release at
#   https://developer.download.nvidia.com/compute/redist/jp/v62/
# For JetPack 6.2 (L4T R36.4) we want torch 2.5.0a0+nv24.08.
# ---------------------------------------------------------------------------
ARG TORCH_INDEX=https://pypi.jetson-ai-lab.dev/jp6/cu126
RUN pip3 install --index-url ${TORCH_INDEX} \
        "torch==2.5.0" "torchvision==0.20.0" \
    || pip3 install \
        --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v62 \
        torch torchvision

# Pin numpy < 2 immediately — the system cv2 4.8 was built against numpy 1.x ABI.
RUN pip3 install "numpy==1.26.4"

# ---------------------------------------------------------------------------
# Stage 2 — Python ML stack (requirements.txt)
# ---------------------------------------------------------------------------
# Copy ONLY requirements.txt first.  This way, editing application code
# doesn't invalidate the (slow) pip-install layer.
# ---------------------------------------------------------------------------
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# ---------------------------------------------------------------------------
# Stage 3 — PaddleOCR (special handling)
# ---------------------------------------------------------------------------
# Paddle 3.x segfaults inside its PIR loader on aarch64; we pin to the stable
# 2.6.2 / 2.7.3 line.  Paddleocr's transitive deps want opencv-contrib-python,
# which would clobber the GStreamer-enabled system cv2 — so we install with
# --no-deps and bring the *needed* deps in explicitly (no opencv among them).
#
# A stub dist-info for opencv-contrib-python is created so PaddleX's metadata
# check passes without actually overwriting system cv2.
# ---------------------------------------------------------------------------
COPY docker/requirements-paddle.txt /tmp/requirements-paddle.txt
RUN pip3 install --no-deps -r /tmp/requirements-paddle.txt && \
    SP=$(python3 -c "import site; print(site.getsitepackages()[0])") && \
    D="$SP/opencv_contrib_python-4.10.0.84.dist-info" && \
    mkdir -p "$D" && \
    printf "Metadata-Version: 2.1\nName: opencv-contrib-python\nVersion: 4.10.0.84\n" > "$D/METADATA" && \
    : > "$D/RECORD" && \
    printf "Wheel-Version: 1.0\nGenerator: manual-stub\nRoot-Is-Purelib: true\nTag: py3-none-any\n" > "$D/WHEEL" && \
    echo "manual" > "$D/INSTALLER"

# ---------------------------------------------------------------------------
# Stage 4 — Prefetch ML models
# ---------------------------------------------------------------------------
# Bundle the heavy weights INSIDE the image so first container start is
# fully offline:
#   - Florence-2-base from HuggingFace (~270 MB)
#   - PaddleOCR PP-OCRv4 en det/rec/cls models (~16 MB)
#   - fast-plate-ocr cct-xs-v1-global-model (~2 MB)
# ---------------------------------------------------------------------------
ENV HF_HOME=/opt/models/huggingface \
    PADDLE_OCR_BASE_DIR=/opt/models/paddleocr \
    FAST_PLATE_OCR_HOME=/opt/models/fast_plate_ocr

COPY docker/prefetch_models.py /tmp/prefetch_models.py
RUN mkdir -p /opt/models && python3 /tmp/prefetch_models.py

# ---------------------------------------------------------------------------
# Stage 5 — Application code
# ---------------------------------------------------------------------------
# Layer ordering: COPY application code LAST so edits don't bust the cache
# for any of the slow steps above.
# ---------------------------------------------------------------------------
WORKDIR /app

# alpr/ holds the YOLO+OCR pipeline (also the .pt weights for engine regen).
COPY alpr/ /app/alpr/

# sparkler/ holds the FastAPI web UI.
COPY sparkler/ /app/sparkler/

# entrypoint + engine regenerator
COPY docker/entrypoint.sh /app/entrypoint.sh
COPY docker/regen_engines.py /app/regen_engines.py
RUN chmod +x /app/entrypoint.sh

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
ENV PYTHONPATH=/app/alpr:/app/sparkler \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    SPARKLER_HOST=0.0.0.0 \
    SPARKLER_PORT=8080 \
    HF_HUB_OFFLINE=0

# Volume mounts (declared as best-practice hints; compose still binds them):
#   /app/sparkler/runs   — SQLite DB + crops + uploads (persist across restarts)
#   /opt/models          — model cache (persist if you want to update weights)
VOLUME ["/app/sparkler/runs", "/opt/models"]

EXPOSE 8080

WORKDIR /app/sparkler
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
