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
# dustynv/l4t-pytorch is a community-maintained base image by NVIDIA
# developer relations.  It bundles a known-good cu126 PyTorch +
# torchvision + numpy + libopenblas + the JetPack 6.x runtime, all
# tested against the actual driver.  Using it eliminates a long tail
# of "torch installs fine at build time but is missing a native lib at
# runtime" surprises that the bare l4t-jetpack base would otherwise
# force us to chase one by one.
FROM dustynv/l4t-pytorch:${L4T_TAG}

# ---------------------------------------------------------------------------
# Stage 0 — supplemental system packages (apt)
# ---------------------------------------------------------------------------
# The dustynv/l4t-pytorch base already ships with python3 + many native
# libs (libopenblas, libcudart, libcudnn, libnvinfer, etc.).  We still
# need:
#   - python3-opencv: SYSTEM OpenCV with GStreamer support, for the
#     IMX219 / IMX477 CSI ribbon cameras.  Pip-installed opencv-python
#     wheels do NOT include GStreamer.
#   - GStreamer plugins (good/bad/ugly) + nvarguscamerasrc helpers.
#   - libgl1 / libglib2.0-0 are runtime deps of OpenCV.
#   - tzdata for sane timestamp printing.
# ---------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-opencv \
        gstreamer1.0-tools gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly gstreamer1.0-libav \
        libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        ffmpeg \
        tzdata ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Stage 1 — pin numpy (belt-and-suspenders)
# ---------------------------------------------------------------------------
# dustynv's base already pins numpy < 2, but a transitive dep from
# requirements.txt could try to bump it.  We force the 1.26.4 line
# explicitly because the system cv2 was built against numpy 1.x ABI.
# ---------------------------------------------------------------------------
RUN pip3 install --force-reinstall --no-deps "numpy==1.26.4"

# ---------------------------------------------------------------------------
# Stage 2 — Python ML stack (requirements.txt)
# ---------------------------------------------------------------------------
# Copy ONLY requirements.txt first.  This way, editing application code
# doesn't invalidate the (slow) pip-install layer.
# ---------------------------------------------------------------------------
COPY docker/requirements.txt /tmp/requirements.txt
COPY docker/constraints.txt /tmp/constraints.txt
RUN pip3 install --constraint /tmp/constraints.txt -r /tmp/requirements.txt

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
RUN pip3 install --no-deps --constraint /tmp/constraints.txt -r /tmp/requirements-paddle.txt && \
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

# Final guard: make absolutely sure numpy is still on the 1.x line
# after every pip install step above.  Cheap belt-and-suspenders.
RUN pip3 install --force-reinstall --no-deps "numpy==1.26.4"

# Final cleanup: forcibly remove any pip-installed OpenCV so the
# GStreamer-enabled system OpenCV (apt python3-opencv, installed in
# Stage 0) is the cv2 that Python sees.  Without this, transitive deps
# from ultralytics / easyocr / fast-plate-ocr pull opencv-python 4.x
# into /usr/local/lib/python3.10/dist-packages/cv2/ which shadows the
# system cv2 and silently breaks CSI camera support (no GStreamer).
RUN pip3 uninstall -y opencv-python opencv-python-headless opencv-contrib-python 2>/dev/null || true

# ---------------------------------------------------------------------------
# Stage 5 — Application code
# ---------------------------------------------------------------------------
# Layer ordering: COPY application code LAST so edits don't bust the cache
# for any of the slow steps above.
# ---------------------------------------------------------------------------
WORKDIR /app

# alpr/ holds the YOLO+OCR pipeline (also the .pt weights for engine regen).
COPY alpr/ /app/alpr/

# sparklers_anpr/ holds the FastAPI web UI.
COPY sparklers_anpr/ /app/sparklers_anpr/

# entrypoint + engine regenerator
COPY docker/entrypoint.sh /app/entrypoint.sh
COPY docker/regen_engines.py /app/regen_engines.py
RUN chmod +x /app/entrypoint.sh

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
ENV PYTHONPATH=/app/alpr:/app/sparklers_anpr \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    SPARKLER_HOST=0.0.0.0 \
    SPARKLER_PORT=8080 \
    HF_HUB_OFFLINE=0

# Volume mounts (declared as best-practice hints; compose still binds them):
#   /app/sparklers_anpr/runs   — SQLite DB + crops + uploads (persist across restarts)
#   /opt/models          — model cache (persist if you want to update weights)
VOLUME ["/app/sparklers_anpr/runs", "/opt/models"]

EXPOSE 8080

WORKDIR /app/sparklers_anpr
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
